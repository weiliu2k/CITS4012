$$W_{ir}=
\begin{cases}
\begin{align*}
&& -0.0930, &&0.0497,\\
&& 0.4670, &&-0.5319,
\end{align*}
\end{cases}
\\
W_{iz}=
\begin{cases}
\begin{align*}
&&-0.6656,&\ \ \ &0.0699,
\\
& &-0.1662,&\ &0.0654,
\end{align*}
\end{cases}
\\
W_{in}=
\begin{cases}
\begin{align*}
& &-0.0449,& &-0.6828,
\\
& &-0.6769,& &-0.1889
\end{align*}
\end{cases}$$


$$
\underbrace{-0.4316,  0.4019}_{b_{ir}}, 
\underbrace{0.1222, -0.4647}_{b_{iz}}, 
\underbrace{-0.5578,  0.4493}_{b_{in}}
$$

LSTM Cell
$$
\begin{align*}
&\color{#82b366}{i(nput\ gate)}&=\color{#82b366}{\sigma(t_{hi}+t_{xi})}
\\
&\color{red}{f(orget\ gate)}&\color{red}{=\sigma(t_{hf}+t_{xf})}
\\
&\color{#0066cc}{o(utput\ gate)}&\color{#0066cc}{=\sigma(t_{ho}+t_{xo})}
\\
&g&=tanh(t_{hg}+t_{xg})
\end{align*}
$$