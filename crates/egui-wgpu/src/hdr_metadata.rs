//! HDR metadata for scRGB surfaces (`VK_EXT_hdr_metadata`).
//!
//! An HDR ([`crate::WgpuConfiguration::prefer_hdr`]) surface on Vulkan is extended linear sRGB,
//! where `1.0` is SDR white. Without HDR metadata, Wayland compositors fall back to the color
//! management protocol's defaults for that transfer function, in which the content's peak
//! equals SDR white (80 cd/m², scRGB `1.0`); `KWin` (since 6.8), for one, then clips everything above `1.0`.
//! [`set_hdr_metadata`] describes how bright the content really gets, and the painter attaches
//! it to every swapchain it creates (Mesa's WSI forwards it to the compositor as
//! `set_mastering_luminance`, `set_max_cll` and `set_max_fall`).
//!
//! Luminances are in the surface's own units: scRGB `1.0` is 80 cd/m², so content that reaches
//! `h` × SDR white has a max content light level of `80 · h`. The compositor maps 80 to its SDR
//! white, so these values follow the content, not the display's absolute brightness.
//!
//! Needs the `hdr-metadata` feature and the Vulkan backend; elsewhere it does nothing.

use core::sync::atomic::{AtomicU64, Ordering};
#[expect(clippy::disallowed_types, reason = "epaint's Mutex::new isn't const, and this is a static")]
use std::sync::Mutex;

/// What an HDR surface's content covers. See the [module docs](self).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrMetadata {
    /// CIE 1931 xy of the mastering display's red, green and blue primaries.
    pub primaries: [[f32; 2]; 3],

    /// CIE 1931 xy of the mastering display's white point.
    pub white_point: [f32; 2],

    /// Mastering display peak, in cd/m² (scRGB `1.0` = 80).
    pub max_luminance: f32,

    /// Mastering display black level, in cd/m².
    pub min_luminance: f32,

    /// Brightest pixel of the content (`MaxCLL`), in cd/m² (scRGB `1.0` = 80).
    pub max_content_light_level: f32,

    /// Brightest frame average of the content (`MaxFALL`), in cd/m².
    pub max_frame_average_light_level: f32,
}

impl HdrMetadata {
    /// Content in BT.709 / sRGB primaries reaching `headroom` × SDR white (scRGB `headroom`).
    pub fn srgb_with_headroom(headroom: f32) -> Self {
        let peak = 80.0 * headroom.max(1.0);
        Self {
            primaries: [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]],
            white_point: [0.3127, 0.3290],
            max_luminance: peak,
            min_luminance: 0.0,
            max_content_light_level: peak,
            max_frame_average_light_level: peak,
        }
    }
}

#[expect(clippy::disallowed_types, reason = "epaint's Mutex::new isn't const, and this is a static")]
static METADATA: Mutex<Option<HdrMetadata>> = Mutex::new(None);
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// Describe the content of HDR surfaces from now on; the painter applies it before the next frame.
///
/// `None` stops sending metadata to new swapchains (Vulkan can't withdraw it from an existing one).
pub fn set_hdr_metadata(metadata: Option<HdrMetadata>) {
    let mut current = METADATA.lock().unwrap_or_else(|e| e.into_inner());
    if *current != metadata {
        *current = metadata;
        GENERATION.fetch_add(1, Ordering::Relaxed);
    }
}

/// The metadata last passed to [`set_hdr_metadata`].
pub fn hdr_metadata() -> Option<HdrMetadata> {
    *METADATA.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bumped by every change in [`set_hdr_metadata`].
pub(crate) fn generation() -> u64 {
    GENERATION.load(Ordering::Relaxed)
}

#[cfg(all(feature = "hdr-metadata", not(target_arch = "wasm32")))]
#[expect(unsafe_code, reason = "wgpu has no HDR metadata API; this goes through wgpu-hal and ash")]
mod vulkan {
    use super::HdrMetadata;
    use ash::vk;
    use wgpu::hal::api::Vulkan;

    const EXTENSION: &core::ffi::CStr = ash::ext::hdr_metadata::NAME;

    /// Create the device with `VK_EXT_hdr_metadata`, when `adapter` is Vulkan and has it.
    ///
    /// `None` means "not applicable": create the device the usual way.
    pub(crate) fn request_device(
        adapter: &wgpu::Adapter,
        desc: &wgpu::DeviceDescriptor<'_>,
    ) -> Option<Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError>> {
        let open = {
            // SAFETY: the hal adapter is only used to open a device on itself, with the
            // features and limits of `desc` and one extra extension it reports as supported.
            let hal_adapter = unsafe { adapter.as_hal::<Vulkan>() }?;
            if !hal_adapter.physical_device_capabilities().supports_extension(EXTENSION) {
                log::debug!("{EXTENSION:?} unsupported: HDR surfaces get no metadata");
                return None;
            }
            let add_extension: Box<wgpu::hal::vulkan::CreateDeviceCallback<'_>> =
                Box::new(|args| args.extensions.push(EXTENSION));
            // SAFETY: the callback only adds a supported extension.
            unsafe {
                hal_adapter.open_with_callback(
                    desc.required_features,
                    &desc.required_limits,
                    &desc.memory_hints,
                    Some(add_extension),
                )
            }
        };
        match open {
            // SAFETY: `open` was created from this adapter with exactly `desc`'s features.
            Ok(open) => Some(unsafe { adapter.create_device_from_hal::<Vulkan>(open, desc) }),
            Err(err) => {
                log::warn!("Vulkan device with {EXTENSION:?} failed ({err}); creating it without");
                None
            }
        }
    }

    /// Attach `metadata` to `surface`'s current swapchain, if it is a Vulkan one on a device
    /// that enabled the extension.
    pub(crate) fn apply(surface: &wgpu::Surface<'_>, device: &wgpu::Device, metadata: &HdrMetadata) {
        // SAFETY: only reads handles; nothing is destroyed or reconfigured.
        let (Some(hal_surface), Some(hal_device)) =
            (unsafe { surface.as_hal::<Vulkan>() }, unsafe { device.as_hal::<Vulkan>() })
        else {
            return;
        };
        if !hal_device.enabled_device_extensions().contains(&EXTENSION) {
            return;
        }
        let Some(swapchain) = hal_surface.raw_native_swapchain() else {
            return;
        };
        let xy = |[x, y]: [f32; 2]| vk::XYColorEXT { x, y };
        let info = vk::HdrMetadataEXT::default()
            .display_primary_red(xy(metadata.primaries[0]))
            .display_primary_green(xy(metadata.primaries[1]))
            .display_primary_blue(xy(metadata.primaries[2]))
            .white_point(xy(metadata.white_point))
            .max_luminance(metadata.max_luminance)
            .min_luminance(metadata.min_luminance)
            .max_content_light_level(metadata.max_content_light_level)
            .max_frame_average_light_level(metadata.max_frame_average_light_level);
        let raw_device = hal_device.raw_device();
        let functions = ash::ext::hdr_metadata::Device::new(hal_device.shared_instance().raw_instance(), raw_device);
        // SAFETY: the swapchain belongs to this device, which enabled the extension.
        unsafe { functions.set_hdr_metadata(&[swapchain], &[info]) };
        log::debug!(
            "HDR metadata: max CLL {:.0}, max FALL {:.0}, mastering {:.0}–{:.4} cd/m²",
            metadata.max_content_light_level,
            metadata.max_frame_average_light_level,
            metadata.max_luminance,
            metadata.min_luminance
        );
    }
}

/// Device creation hook for HDR surfaces; `None` = create the device the usual way.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn request_device(
    _adapter: &wgpu::Adapter,
    _desc: &wgpu::DeviceDescriptor<'_>,
) -> Option<Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError>> {
    #[cfg(feature = "hdr-metadata")]
    let device = vulkan::request_device(_adapter, _desc);
    #[cfg(not(feature = "hdr-metadata"))]
    let device = None;
    device
}

/// Attach the current metadata (if any) to `surface`'s swapchain, when it is an HDR one.
pub(crate) fn apply(_surface: &wgpu::Surface<'_>, _device: &wgpu::Device, _format: wgpu::TextureFormat) {
    #[cfg(all(feature = "hdr-metadata", not(target_arch = "wasm32")))]
    if _format == wgpu::TextureFormat::Rgba16Float
        && let Some(metadata) = hdr_metadata()
    {
        vulkan::apply(_surface, _device, &metadata);
    }
}
