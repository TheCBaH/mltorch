(** CLIP's image preprocessing for a raw RGB image: resize the shorter edge to
    [size] (bicubic, Pillow's arithmetic), take the central [size] x [size]
    crop, scale to [0, 1] and normalize per channel. The result is
    channels-first, as the model takes it. *)

val tensor :
  Ppm.t ->
  filter:Resample.filter ->
  resize:int ->
  crop:int ->
  flip:bool ->
  mean:float array ->
  std:float array ->
  float array
(** The general form: the shorter edge to [resize] with [filter], a central
    [crop] x [crop] window, values scaled by 1/255 and normalized
    ([(v - mean) / std] per channel; use mean 0 and std 1 for none), and with
    [flip] the channel order reversed (RGB to BGR) before they are laid out
    channels-first. *)

val pixel_values :
  Ppm.t -> size:int -> mean:float array -> std:float array -> float array
(** [3 * size * size] values, channel-major, each rounded to float32. *)

val resized_size : width:int -> height:int -> size:int -> int * int
(** The image size after the shorter edge is brought to [size]: the longer edge
    is [int (size * long / short)], truncated. *)
