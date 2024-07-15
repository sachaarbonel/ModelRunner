use std::sync::OnceLock;

use candle_core::{utils::{cuda_is_available, metal_is_available}, Device};


/// Create a candle device that uses any available accelerator.
pub fn accelerated_device_if_available() -> candle_core::Result<Device> {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    if let Some(device) = DEVICE.get() {
        return Ok(device.clone());
    }
    let device = if cuda_is_available() {
        Device::new_cuda(0)?
    } else if metal_is_available() {
        Device::new_metal(0)?
    } else {
        #[cfg(all(debug_assertions, target_os = "macos", target_arch = "aarch64"))]
        {
            println!("Running on CPU, to run on GPU(metal), build with `--features metal`");
        }
        #[cfg(not(all(debug_assertions, target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build with `--features cuda`");
        }
        Device::Cpu
    };
    let _ = DEVICE.set(device.clone());
    Ok(device)
}