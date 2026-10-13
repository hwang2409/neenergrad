# backprop

notes taken while reading [christopher olah's 2015 blog](https://colah.github.io/posts/2015-08-Backprop/)

## computational graphs
consider the expression $e = (a+b) * (b+1)$

in total, there are 3 ops: $(a+b)$, $(b+1)$ and the product of these 2 intermediate results

let: $c = (a * b)$ and $d = (b+1)$, we now have $e = c*d$

```mermaid
flowchart LR
A(["a"])
B(["b"])
A --> C["c=a+b"]
B --> C
B --> D["d=b+1"]
C --> E["e=c*d"]
D --> E
```
