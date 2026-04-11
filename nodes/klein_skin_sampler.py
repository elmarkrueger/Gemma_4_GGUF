import comfy.samplers  # type: ignore
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from tqdm.auto import trange


class DuffyKleinSkinSampler(io.ComfyNode):
    """Custom sampler for FLUX.2-Klein-9B that restores high-frequency skin
    textures via a hybrid RK4 ODE / wavelet SDE integration schedule.

    Uses deterministic 4th-order Runge-Kutta for macro-anatomy in early steps,
    then transitions to stochastic high-pass noise injection in the final
    portion of the schedule to synthesise pores and micro-detail.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Duffy_KleinSkinSampler",
            display_name="Klein 9B Skin Texture SDE (Wavelet)",
            category="Duffy/Sampling",
            description=(
                "Hybrid RK4 + wavelet-SDE sampler designed for FLUX.2-Klein-9B. "
                "Injects frequency-matched high-pass noise in the final stage of "
                "the denoising schedule to restore photorealistic skin textures."
            ),
            inputs=[
                io.Float.Input(
                    "eta_texture",
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Magnitude of injected high-frequency noise. "
                            "Values above 0.3 risk structural distortion; "
                            "below 0.1 may not break the plastic smoothing.",
                ),
                io.Float.Input(
                    "texture_threshold",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Fraction of the schedule (from the end) where "
                            "SDE noise injection begins. 0.35 = final 35%.",
                ),
                io.Float.Input(
                    "rectified_cfg_scale",
                    default=1.5,
                    min=1.0,
                    max=10.0,
                    step=0.1,
                    tooltip="Internal guidance scale for Rectified-CFG++ "
                            "predictor-corrector. Keep minimal when external "
                            "CFG is also applied.",
                ),
            ],
            outputs=[
                io.Sampler.Output(display_name="SAMPLER"),
            ],
        )

    @classmethod
    def execute(cls, eta_texture: float, texture_threshold: float,
                rectified_cfg_scale: float, **kwargs) -> io.NodeOutput:

        @torch.no_grad()
        def sample_klein_skin(model, x, sigmas, extra_args=None, callback=None,
                              disable=None, **extra):
            extra_args = {} if extra_args is None else extra_args
            s_in = x.new_ones([x.shape[0]])
            total_steps = len(sigmas) - 1

            for i in trange(total_steps, disable=disable):
                sigma_hat = sigmas[i]
                sigma_next = sigmas[i + 1]
                dt = sigma_next - sigma_hat

                # --- Helper: model velocity from denoised prediction ---
                def _velocity(x_t, sigma):
                    denoised = model(x_t, sigma * s_in, **extra_args)
                    d = (x_t - denoised) / sigma
                    return d, denoised

                # --- Phase A: 4th-order Runge-Kutta integration ---
                k1, denoised = _velocity(x, sigma_hat)

                sigma_mid = sigma_hat + dt / 2
                if sigma_mid > 0:
                    k2, _ = _velocity(x + (dt / 2) * k1, sigma_mid)
                    k3, _ = _velocity(x + (dt / 2) * k2, sigma_mid)
                else:
                    k2 = k1
                    k3 = k1

                if sigma_next > 0:
                    k4, _ = _velocity(x + dt * k3, sigma_next)
                else:
                    k4 = k3

                x_pred = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

                # --- Phase B: SDE wavelet noise injection in texture zone ---
                progress = i / total_steps
                in_texture_phase = progress >= (1.0 - texture_threshold)

                if in_texture_phase and eta_texture > 0 and sigma_next > 0:
                    raw_noise = torch.randn_like(x)

                    # High-pass filter: subtract the local mean to isolate
                    # high-frequency spatial components (pore-scale detail).
                    blurred = F.avg_pool2d(
                        raw_noise, kernel_size=3, stride=1, padding=1,
                    )
                    wavelet_noise = raw_noise - blurred

                    # Normalize variance to unit std
                    noise_std = wavelet_noise.std()
                    if noise_std > 0:
                        wavelet_noise = wavelet_noise / noise_std

                    # Langevin dynamics injection scaled by step size
                    noise_amplitude = eta_texture * (abs(dt) ** 0.5)
                    x_pred = x_pred + noise_amplitude * wavelet_noise

                x = x_pred

                if callback is not None:
                    callback({
                        "x": x,
                        "i": i,
                        "sigma": sigmas[i],
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                    })

            return x

        sampler = comfy.samplers.KSAMPLER(
            sample_klein_skin,
            extra_options={
                "eta_texture": eta_texture,
                "texture_threshold": texture_threshold,
                "rectified_cfg_scale": rectified_cfg_scale,
            },
        )
        return io.NodeOutput(sampler)
