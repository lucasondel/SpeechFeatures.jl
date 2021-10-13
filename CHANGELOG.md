# Releases

## 0.3.2
* Avoid GC pressure by doing most of the operations in `stft` and
  `mfcc` in-place.

## 0.3.1
* Renamed `FilterBank` to `filterbank` to homogenize the user interface.
  The previous `FilterBank` function is still exported but is mark
  as deprecated.

## 0.3.0
* Simplfied the code and refactored the user interface.
* Added Pluto notebook examples.

## 0.2.0
* The output features are in matrix form insteaad of array of arrays.

## 0.1.0
* initial release
