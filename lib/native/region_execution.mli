type t = Pixel_loop of Expr.Value.t | Region_loop of Region_program.t

val lower : Region_program.t -> t
