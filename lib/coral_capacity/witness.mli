(** The two images the segment plan's capacity claim rests on, with sizes
    derived from the shapes. [code] is the executable size the caller measured;
    the data sizes are the images' own. *)

val pointwise_mobilenetv2_050 : code:Code_bytes.t -> Image.t
(** Pointwise 8 to 48 at 112x112 with its Hardtanh, blocks of two rows: input
    401,408, output 2,408,448, scratch 43,008. *)

val expand_depthwise_efficientnet_b0 : code:Code_bytes.t -> Image.t
(** Expand 16 to 96, Silu, depthwise 3x3 stride 2, Silu: input 802,816, output
    1,204,224, a band of four rows 172,032, scratch 86,016. *)

val all : code:Code_bytes.t -> Image.t list
