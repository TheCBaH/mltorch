(** CLIP's image preprocessing for a raw RGB image: resize the shorter edge to
    [size] (bicubic, Pillow's arithmetic), take the central [size] x [size]
    crop, scale to [0, 1] and normalize per channel. The result is
    channels-first, as the model takes it. *)

val pixel_values :
  Ppm.t -> size:int -> mean:float array -> std:float array -> float array
(** [3 * size * size] values, channel-major, each rounded to float32. *)

val resized_size : width:int -> height:int -> size:int -> int * int
(** The image size after the shorter edge is brought to [size]: the longer edge
    is [int (size * long / short)], truncated. *)
