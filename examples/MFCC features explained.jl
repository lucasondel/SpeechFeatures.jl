### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 75ae3354-2aaa-11ec-1805-d1efd04acf08
begin
	using BenchmarkTools
	using LinearAlgebra
	using Plots
	using PlutoUI
	using Pkg
	using WAV
	Pkg.activate("/home/ondel/Repositories/SpeechFeatures.jl")
	Pkg.instantiate()
	using SpeechFeatures
	using FFTW
	using PaddedViews
end

# ╔═╡ f87589e3-c5d7-41b5-b376-bcf9eec006d1
md"""
# MFCC features explained
*[Lucas Ondel](https://lucasondel.github.io/), October 2021*

In this notebook, we show step-by-step how to compute the Mel-Frequency Cepstral Coefficient (MFCC) features.
"""

# ╔═╡ d9026f53-756d-4862-a258-f9663a9a76a2
md"""
We will use mostly two packages: [WAV.jl](https://github.com/dancasimiro/WAV.jl) to load the WAV data and [SpeechFeatures.jl](https://github.com/lucasondel/SpeechFeatures.jl) which provides utility function to extract the features. 
"""

# ╔═╡ 319b69f9-6c9d-4d22-9896-055800cf5de8
TableOfContents()

# ╔═╡ 844d4433-bc74-472b-9723-d4136bf56f0f
md"""
## Loading the data 

We will work with a sample from the TIMIT corpus freely available on the LDC website. First, we download it in the directory of this notebook.
"""

# ╔═╡ 86a58676-7f23-4e45-8ffb-0413e00e3237
wavpath = download("https://catalog.ldc.upenn.edu/desc/addenda/LDC93S1.wav",
		 		   joinpath(@__DIR__, "sample.wav"))

# ╔═╡ f6647baa-e24a-4c67-9c1c-ae95cd9239e4
md"""
Now, we read it using the `wavread` function. This function returns the channel matrix of size $S \times C$ and the sampling frequency.
"""

# ╔═╡ 4cd9e50b-6e12-48e0-812d-00af1598b32c
channels, fs = wavread(wavpath; format="double")

# ╔═╡ f2227028-3926-4864-9330-33cacc6349be
md"""
Each channel corresponds to the recording of one microphone. Concretely, if you have a mono recording you will have one channel, if you have stereo recording you will have 2 channels, etc.

Here, we only take the first channel (the data is mono anyway).
"""

# ╔═╡ ab6e2ce4-5941-4441-ae1d-7417a9b2b84e
x = channels[:, 1]

# ╔═╡ 786d833c-4a58-48d3-9e6e-b7869fd02a2e
md"""
Now we can plot the waveform.
"""

# ╔═╡ 8d116895-703f-4fd5-a3a9-aa8925ef7461
plot((1:length(x)) ./ fs, x; xlabel="Time (s)", legend=false)

# ╔═╡ 8daea702-d679-4ef0-96d5-230f597889a6
md"""
## Dithering

To avoid having frequency components with 0 energy, we add a tiny bit of white noise to the signal.
"""

# ╔═╡ db90b23f-d363-432d-a2e2-5772bf1657ba
dithering = 1e-12

# ╔═╡ 0a9c2db4-bd6e-42e5-874f-28f75b5385c5
x .+= randn(length(x)) .* dithering

# ╔═╡ 0a2780df-8fee-4b27-a944-3e0c7f2aa053
md"""
## DC removal

In general, most signals will be centered around zero. However, if this is not the case, this can introduce an undesired bias. To avoid this, we simply remove the Direct Component (DC) as a first step. 
"""


# ╔═╡ da662210-d760-4989-b6c3-99c58395514f
x .-= mean(x)

# ╔═╡ 8bbd7a37-c714-4f64-81d0-48a18717336b
md"""
## Getting frames

The MFCC features are based on short-term spectral representation of the signal. Therefore, we need to divide the signal into overlapping frames that will be later transformed to the spectral domain. 

The frame extraction has 2 parameters:
  * the frame duration
  * the step, i.e. the time between the beginning of two adjacent frames
"""

# ╔═╡ 70ef4159-f09a-4e2d-a266-c86972a6a611
frameduration = 0.025 # in second

# ╔═╡ ce94d7c3-5814-4805-a5c5-bf6e56c412ff
framestep = 0.01 # in second

# ╔═╡ b457c84c-50aa-43aa-84d6-d38cff22883b
md"""
To get the size of a frame in terms of number of samples, we multiply the frame duration and the sampling frequency.
"""

# ╔═╡ 2623d5d5-ba06-4247-8929-5d98d8c65c89
framesize = Int64(frameduration * fs)

# ╔═╡ 65e7fb61-4fb5-4826-8487-2c613b782773
X = hcat(SpeechFeatures.eachframe(x; srate=fs, frameduration, framestep)...)

# ╔═╡ 045b825e-47ed-462f-912d-3954812651a8
md"""
## Pre-emphasis

The purpose of this step is to increase the dynamic range of the high-frequency components. The pre-emphasis filter is defined as:
```math
y[n] = x[n] - k \cdot x[n-1],
```
where $k \in [0, 1]$. In general, we set $k = 0.97$.
"""

# ╔═╡ de44637f-2f24-4a5a-b1f3-1f5dd90f85af
function preemphasis(x; k = 0.97)
	y = similar(x)
	y[1] = x[1]
	prev = x[1]
	for i in 2:length(x)
		y[i] = x[i] - k*prev
		prev = x[i]
	end
	y
end

# ╔═╡ 8f876646-f9bf-489a-8002-607d38eee4e9
prmX = hcat(map(preemphasis, eachcol(X))...)

# ╔═╡ d49fd625-0415-44b4-94a9-94d2780aa0c3
md"""
## Windowing

Each frame, will be multiplied by a windowing function. Here we compare different type of windows.
"""

# ╔═╡ dfa1065d-29ad-443e-930c-33ae740652d7
# The exponent is used in the Kaldi features extraction. 
# The idea is to get a Hamming-like window which goes to 0
# at the edge.
hann = SpeechFeatures.HannWindow(framesize) .^ 0.85

# ╔═╡ b1b54b9e-bca0-4e65-901d-6cf690331e2c
hamming = SpeechFeatures.HammingWindow(framesize) 

# ╔═╡ 9fd33feb-e0c2-45fc-80a2-baa9fe9bbcd3
rectangular = SpeechFeatures.RectangularWindow(framesize) 

# ╔═╡ 717bf350-954e-4e33-95ef-063c89fe90ae
begin 
	plot((1:framesize) ./ fs, hann; linewidth=2, yrange=(0,1), label = "Hann")
	plot!((1:framesize) ./ fs, hamming; linewidth=2, yrange=(0,1), label = "Hamming")
	plot!((1:framesize) ./ fs, rectangular; linewidth=2, yrange=(0,1), label = "Rectangular")
end

# ╔═╡ 19148014-f27e-4821-946e-fb68345a7641
md"""
Change the line below to select the window.
"""

# ╔═╡ f283f94f-993a-4156-b606-8014aae341ca
window = hann

# ╔═╡ 95b9d153-4934-45d8-b9f3-138d93757bfb
md"""
Finally, we multiply (in-place) the window on each frame.
"""

# ╔═╡ f4f33068-88f2-4b23-bb7b-47abc9e34bac
wX = hcat(map(x -> x .* window, eachcol(prmX))...)

# ╔═╡ e2b3d74c-9199-4c03-8405-fb19f171fd05
md"""
## Short-term spectrum

Now, we have compute the Fourier transform for each fame.

For efficiency reason, it is common that we compute the Fast Fourier Transform (FFT) on vector of length being a power of two. For this, we simply take the first power of two larger than the current frame size.
"""

# ╔═╡ d4cb69e2-fd46-4bc6-ae1b-8e041e015f76
fftlen = Int64(2^ceil(log2(size(wX, 1)))) 

# ╔═╡ d18010a6-6ac5-493f-92fc-590bf6bd6fe3
md"""
We pad the frames with zeros at the end to match the size of the FFT.
"""

# ╔═╡ 3ae4936f-faa9-45ac-90cd-c2a1bc782550
pX = PaddedView(0, wX, (fftlen, size(wX, 2)))[1:end,:]

# ╔═╡ 831accf5-fa88-492c-9a9e-6e7e58a6ce91
md"""
Get the log-magnitude spectrum of each frame.
"""

# ╔═╡ 68ef4c2f-eb3c-458f-96ad-a301754bc469
S = abs.(rfft(pX, 1)[1:end-1,:])

# ╔═╡ 013a0118-8181-461e-bc8f-fb6479787383
heatmap((1:size(S,2)) ./ 100, (fs/2) .* (0:size(S,1)-1) ./ size(S,1), S;
		xlabel="time (s)", ylabel="frequency (Hz)")

# ╔═╡ 6c94c82f-837b-4bcc-8db8-79ad8a0382d4
md"""
## Filter bank

The short-term spectrum we have extracted is useful but it is not very faithful of what humans actually perceive. Actually, our spectral resolution is much lower maps non-linearly with the frequency range.  

The mel-scale is an approximation of the human frequency-perception. It is given by:
```math
m = 1127 \ln(1 + \frac{f}{700})
```
"""

# ╔═╡ 5de5c24d-5407-4001-96a1-21094719c65f
plot(0.1:0.1:8000, SpeechFeatures.freq2mel.(0.1:0.1:8000); 
	 xlabel="Hertz", ylabel="Mel", legend=false)

# ╔═╡ 6398bf0b-295e-4c6d-a9fa-0df8c1bd2807
md"""
We create a set of 26 filters whose centers are equally spaced on the mel-scale. 
"""

# ╔═╡ 20c2333c-f368-4077-86ef-794e849adb0a
fbank = SpeechFeatures.FilterBank(26; srate=fs, fftlen, lofreq=20, hifreq=7600)

# ╔═╡ 406bce61-409c-4d3e-8d50-930f4b48387b
plot((fs/2) .* (1:size(fbank,2)) ./ size(fbank,2), fbank';
	 size=(800, 400), legend=false, xlabel="frequency (Hz)")

# ╔═╡ df2ee608-be7f-44d6-bab8-41a67fbe9e48
md"""
Applying the filter bank in the spectral domain amounts to multily the frame matrix with the filter bank matrix.
"""

# ╔═╡ 39b9c15e-8459-45a3-b4de-9bada6203580
fS = fbank * S 

# ╔═╡ f19ba4e7-8625-43b9-a989-95e3f7ab1825
heatmap((1:size(fS,2)) ./ 100, 1:size(fS,1), fS;
		xlabel="time (s)")

# ╔═╡ 664a2a9b-d12a-4230-b0bb-c4eb32dbd253
md"""
Furthermore, we get our spectrum in the log-domain to reduce the dynamic range.
"""

# ╔═╡ 07c8b5e6-0f95-49ea-8fb8-1aa24c6bc11c
lS = log.(fS)

# ╔═╡ 82550777-784f-4c97-86e2-1e0bad53f9ae
heatmap((1:size(fS,2)) ./ 100, 1:size(fS,1), lS;
		xlabel="time (s)")

# ╔═╡ b0f01b5a-40a8-4bc1-bab6-ad9ea1daff73
md"""
## DCT

We can decorrelate and reduce the dimension of the the features by applying a Discrete  Cosine Transform (DCT). By doing so, our features are now in the "cepstral" domain.
"""

# ╔═╡ ce9a8367-48d0-4947-8499-50b674d763ea
nceps = 13

# ╔═╡ d2c573f1-58b2-4104-b619-56cfbb522063
C = dct(lS, 1)[1:nceps,:]

# ╔═╡ 057c98a6-9878-4d51-aede-f77603af7e16
heatmap((1:size(C,2)) ./ 100, 1:size(C,1), C;
		xlabel="time (s)")

# ╔═╡ 09528ac5-ffda-4d0a-b7ce-522722593644
md"""
## Liftering

Now we "lifter" (i.e. filtering in the cepstral domain) to even the dynamic ranges across cepstral coefficients.
"""

# ╔═╡ 9c9d7293-68c7-4d66-bea0-1743019bf9dc
function makelifter(N, L)
	t = Vector(1:N)
	1 .+ L/2 * sin.(π * t / L)
end

# ╔═╡ f97d744f-be53-42a4-8800-e83d4440b0e6
lifter = makelifter(size(C,1), 22)

# ╔═╡ 857e1fa9-6997-4c96-90fa-ae0fbb9e8cc2
plot(lifter, legend=false)

# ╔═╡ 83af226d-f60f-461e-8c28-835160d5c270
lC = hcat(map(x -> x .* lifter, eachcol(C))...)

# ╔═╡ b8180e1c-d698-44ec-9372-a7d8f133b3f1
heatmap((1:size(lC,2)) ./ 100, 1:size(lC,1), lC;
		xlabel="time (s)")

# ╔═╡ 0582ef6f-dcd2-42c2-a1bd-8aac011cf166
md"""
## Dynamic features

Finally, we add the first and second derivatives to the signal. The derivatives are
calculated as:
```math
\dot{x} \approx \frac{\sum_{k=1}^K k \cdot (x[n+k] - x[n-k]) }{2\sum_{k=1}^K k^2}
```
"""

# ╔═╡ 09ede491-cb56-4327-b2e0-6e10b3a5483d
Δ = SpeechFeatures.delta(C, 2)

# ╔═╡ 9c0fd83e-9217-4516-a4e6-9566a7e78b31
ΔΔ = SpeechFeatures.delta(Δ, 2)

# ╔═╡ af564d77-dc11-4125-bde3-1f07c4521937
features = vcat(C, Δ, ΔΔ)

# ╔═╡ Cell order:
# ╟─f87589e3-c5d7-41b5-b376-bcf9eec006d1
# ╟─d9026f53-756d-4862-a258-f9663a9a76a2
# ╠═75ae3354-2aaa-11ec-1805-d1efd04acf08
# ╠═319b69f9-6c9d-4d22-9896-055800cf5de8
# ╟─844d4433-bc74-472b-9723-d4136bf56f0f
# ╠═86a58676-7f23-4e45-8ffb-0413e00e3237
# ╟─f6647baa-e24a-4c67-9c1c-ae95cd9239e4
# ╠═4cd9e50b-6e12-48e0-812d-00af1598b32c
# ╟─f2227028-3926-4864-9330-33cacc6349be
# ╠═ab6e2ce4-5941-4441-ae1d-7417a9b2b84e
# ╟─786d833c-4a58-48d3-9e6e-b7869fd02a2e
# ╠═8d116895-703f-4fd5-a3a9-aa8925ef7461
# ╟─8daea702-d679-4ef0-96d5-230f597889a6
# ╠═db90b23f-d363-432d-a2e2-5772bf1657ba
# ╠═0a9c2db4-bd6e-42e5-874f-28f75b5385c5
# ╟─0a2780df-8fee-4b27-a944-3e0c7f2aa053
# ╠═da662210-d760-4989-b6c3-99c58395514f
# ╟─8bbd7a37-c714-4f64-81d0-48a18717336b
# ╠═70ef4159-f09a-4e2d-a266-c86972a6a611
# ╠═ce94d7c3-5814-4805-a5c5-bf6e56c412ff
# ╟─b457c84c-50aa-43aa-84d6-d38cff22883b
# ╠═2623d5d5-ba06-4247-8929-5d98d8c65c89
# ╠═65e7fb61-4fb5-4826-8487-2c613b782773
# ╟─045b825e-47ed-462f-912d-3954812651a8
# ╠═de44637f-2f24-4a5a-b1f3-1f5dd90f85af
# ╠═8f876646-f9bf-489a-8002-607d38eee4e9
# ╟─d49fd625-0415-44b4-94a9-94d2780aa0c3
# ╟─dfa1065d-29ad-443e-930c-33ae740652d7
# ╠═b1b54b9e-bca0-4e65-901d-6cf690331e2c
# ╠═9fd33feb-e0c2-45fc-80a2-baa9fe9bbcd3
# ╠═717bf350-954e-4e33-95ef-063c89fe90ae
# ╟─19148014-f27e-4821-946e-fb68345a7641
# ╠═f283f94f-993a-4156-b606-8014aae341ca
# ╟─95b9d153-4934-45d8-b9f3-138d93757bfb
# ╠═f4f33068-88f2-4b23-bb7b-47abc9e34bac
# ╟─e2b3d74c-9199-4c03-8405-fb19f171fd05
# ╠═d4cb69e2-fd46-4bc6-ae1b-8e041e015f76
# ╟─d18010a6-6ac5-493f-92fc-590bf6bd6fe3
# ╠═3ae4936f-faa9-45ac-90cd-c2a1bc782550
# ╟─831accf5-fa88-492c-9a9e-6e7e58a6ce91
# ╠═68ef4c2f-eb3c-458f-96ad-a301754bc469
# ╠═013a0118-8181-461e-bc8f-fb6479787383
# ╟─6c94c82f-837b-4bcc-8db8-79ad8a0382d4
# ╠═5de5c24d-5407-4001-96a1-21094719c65f
# ╟─6398bf0b-295e-4c6d-a9fa-0df8c1bd2807
# ╠═20c2333c-f368-4077-86ef-794e849adb0a
# ╠═406bce61-409c-4d3e-8d50-930f4b48387b
# ╟─df2ee608-be7f-44d6-bab8-41a67fbe9e48
# ╠═39b9c15e-8459-45a3-b4de-9bada6203580
# ╠═f19ba4e7-8625-43b9-a989-95e3f7ab1825
# ╟─664a2a9b-d12a-4230-b0bb-c4eb32dbd253
# ╠═07c8b5e6-0f95-49ea-8fb8-1aa24c6bc11c
# ╠═82550777-784f-4c97-86e2-1e0bad53f9ae
# ╟─b0f01b5a-40a8-4bc1-bab6-ad9ea1daff73
# ╠═ce9a8367-48d0-4947-8499-50b674d763ea
# ╠═d2c573f1-58b2-4104-b619-56cfbb522063
# ╠═057c98a6-9878-4d51-aede-f77603af7e16
# ╟─09528ac5-ffda-4d0a-b7ce-522722593644
# ╠═9c9d7293-68c7-4d66-bea0-1743019bf9dc
# ╠═f97d744f-be53-42a4-8800-e83d4440b0e6
# ╠═857e1fa9-6997-4c96-90fa-ae0fbb9e8cc2
# ╠═83af226d-f60f-461e-8c28-835160d5c270
# ╠═b8180e1c-d698-44ec-9372-a7d8f133b3f1
# ╟─0582ef6f-dcd2-42c2-a1bd-8aac011cf166
# ╠═09ede491-cb56-4327-b2e0-6e10b3a5483d
# ╠═9c0fd83e-9217-4516-a4e6-9566a7e78b31
# ╠═af564d77-dc11-4125-bde3-1f07c4521937
