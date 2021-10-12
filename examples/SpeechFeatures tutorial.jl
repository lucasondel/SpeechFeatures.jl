### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 3d4495cd-3107-4855-8a57-b4154f7af653
begin 
	using Pkg
	Pkg.activate("/home/ondel/Repositories/SpeechFeatures.jl")
	Pkg.instantiate()
	using SpeechFeatures
	using WAV
	using Plots
end

# ╔═╡ 9a0f7ed6-2b6c-11ec-23c2-7fd783169875
md"""
# SpeechFeatures tutorial
*[Lucas Ondel](https://lucasondel.github.io/), October 2021*

This is notebook shows how to use the [SpeechFeatures](https://github.com/lucasondel/SpeechFeatures.jl) Julia package.
"""

# ╔═╡ 160a988a-20cb-4ff6-a6a6-7ace16f4a97a
md"""
## Loading the data 

We will work with a sample from the TIMIT corpus freely available on the LDC website. First, we download it in the directory of this notebook.
"""

# ╔═╡ 332e11ed-69ed-41a3-a1f1-4f718efdffa1
wavpath = download("https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav",
		 		   joinpath(@__DIR__, "sample.wav"))

# ╔═╡ cf2a2daa-3334-45d3-83fb-132c0c211d96
channels, fs = wavread(wavpath, format="double")

# ╔═╡ 4f43424d-4d92-4a47-8792-cf0e3a448292
x = channels[:,1]

# ╔═╡ baaed408-18f7-4648-a04c-99c58abd907f
md"""
## Short-Term Fourier Transform

The first step of any speech features is to get the FFT of overlapping frames. 
"""

# ╔═╡ bf913df5-348a-4a3e-8c96-e52dcceceb43
S, fftlen = stft(x; srate=fs)

# ╔═╡ 2d9db55a-4826-4173-a3b2-998187803ad2
heatmap(log.(abs.(S)))

# ╔═╡ 83d7af4e-dddf-4621-81fb-4a5e11c1dcf0
md"""
## Mel-spectrum

To the the mel-spectrum we create the filter bank and apply on the stft.
"""

# ╔═╡ fe42604a-b264-43ec-abab-135fd13c326c
fbank = FilterBank(26; fftlen)

# ╔═╡ 303a712f-7048-4a55-a938-e92c0d396bfd
melspectrum = fbank * abs.(S)

# ╔═╡ 575d5378-6fc1-4334-8975-913288136aa5
heatmap(log.(melspectrum))

# ╔═╡ d399c618-95e1-43c3-a9f4-1ce75e41f215
md"""
## MFCC

The mel-cepstral coefficients can be obtained from the mel-cepstrum by calling `mfcc`
"""

# ╔═╡ 8b3961a5-954d-4b05-8701-f21bed104d7a
MFCCs = mfcc(melspectrum; nceps=13)

# ╔═╡ e705a092-6e9d-45c8-9194-dbbbd2fa61b7
heatmap(MFCCs)

# ╔═╡ cc5a1be6-88bc-421a-a064-5f3bfa16e21d
md"""
## Derivatives 
Finally, to get the first and second derivatives use `add_deltas`.
"""

# ╔═╡ 1031d2f5-18ec-44fc-b10f-640954991b31
MFCCs_Δ_ΔΔ = add_deltas(MFCCs; order=2)

# ╔═╡ 0d535c94-4a2f-4227-9c62-d757f94f581e
size(MFCCs_Δ_ΔΔ)

# ╔═╡ 84c10764-0d44-43d0-8c31-e34f33df5496
heatmap(MFCCs_Δ_ΔΔ)

# ╔═╡ Cell order:
# ╟─9a0f7ed6-2b6c-11ec-23c2-7fd783169875
# ╠═3d4495cd-3107-4855-8a57-b4154f7af653
# ╟─160a988a-20cb-4ff6-a6a6-7ace16f4a97a
# ╠═332e11ed-69ed-41a3-a1f1-4f718efdffa1
# ╠═cf2a2daa-3334-45d3-83fb-132c0c211d96
# ╠═4f43424d-4d92-4a47-8792-cf0e3a448292
# ╟─baaed408-18f7-4648-a04c-99c58abd907f
# ╠═bf913df5-348a-4a3e-8c96-e52dcceceb43
# ╠═2d9db55a-4826-4173-a3b2-998187803ad2
# ╟─83d7af4e-dddf-4621-81fb-4a5e11c1dcf0
# ╠═fe42604a-b264-43ec-abab-135fd13c326c
# ╠═303a712f-7048-4a55-a938-e92c0d396bfd
# ╠═575d5378-6fc1-4334-8975-913288136aa5
# ╟─d399c618-95e1-43c3-a9f4-1ce75e41f215
# ╠═8b3961a5-954d-4b05-8701-f21bed104d7a
# ╠═e705a092-6e9d-45c8-9194-dbbbd2fa61b7
# ╟─cc5a1be6-88bc-421a-a064-5f3bfa16e21d
# ╠═1031d2f5-18ec-44fc-b10f-640954991b31
# ╠═0d535c94-4a2f-4227-9c62-d757f94f581e
# ╠═84c10764-0d44-43d0-8c31-e34f33df5496
