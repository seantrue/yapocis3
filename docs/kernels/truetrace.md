# Kernels

## `truetrace`

### Interface specification

```
interface truetrace {
    # Initialize jump-flood seeds and distance from a binary mask
    kernel init_seed_from_mask(width_of mask, height_of mask,
                               in int mask,
                               out float seed,
                               out float dist);

    # One JFA step with given jump length; updates distance, writes next seeds
    kernel jfa_step(width_of seed_in, height_of seed_in,
                    int jump,
                    in float seed_in,
                    inout float dist,
                    out float seed_out);

    # Marching squares over distance field to emit line segments at iso
    kernel marching_squares(width_of dist, height_of dist,
                            in float dist,
                            float iso,
                            inout uint32 seg_counter,
                            out float segments_out);
};
```

