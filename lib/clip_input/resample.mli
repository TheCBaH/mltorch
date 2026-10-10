(** Pillow's bicubic [Image.resize] on 8-bit RGB, in the fixed-point form its C
    resampler uses: per-axis coefficients from the filter, normalized to sum to
    one, scaled to 22 fractional bits and rounded half away from zero, applied
    horizontally and then vertically with each pass rounded back to 8 bits. The
    two passes and the rounding between them are why a float resize is not the
    same image. *)

type filter = Bicubic | Bilinear

val resize : filter -> Ppm.t -> width:int -> height:int -> Ppm.t
(** Pillow's [Image.resize] with [Image.BICUBIC] or [Image.BILINEAR]. *)

val bicubic : Ppm.t -> width:int -> height:int -> Ppm.t
(** The image resized to exactly [width] x [height]; an axis whose size is
    unchanged is not resampled, as in Pillow. *)
