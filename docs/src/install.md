# Installation

## Installation of Julia

The SpeechFeatures package was developped with Julia 1.4.1.
If you haven't installed Julia already, follow the instruction
[here](https://julialang.org/downloads/)

!!! tip
    It is a common practice in Julia to use non-ascii characters while
    coding such as greek letters or mathemtacal symbols. We highly
    recommend to add Julia support to your editor. Plugin for
    [vim](https://www.vim.org/)/[neovim](https://neovim.io/)
    and [emacs](https://www.gnu.org/software/emacs/) can be found
    [here](https://github.com/JuliaEditorSupport).

## Installation of SpeechFeatures

In the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/)
prompt, press `]` to enther the Pkg REPL and then type:
```
(@v1.4) pkg> add https://github.com/BUTSpeechFIT/SpeechFeatures
```

This will install the package along with its dependencies into your
Julia installation.
Note that `(@v1.4) pkg>` is simply the prompt message, you should not
type it !

