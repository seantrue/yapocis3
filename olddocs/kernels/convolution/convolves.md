# Kernels

## `convolves`
* convolves.interface has unprocessed template markup.

```
interface convolves {
%for name,conv in convs:
        kernel ${name}(sizeof a, in float a, outlike a);
%endfor 
};
```

