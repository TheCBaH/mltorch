type t = Pixel_loop of Expr.Value.t | Region_loop of Region_program.t

let lower program =
  match Region_program.pixel_expression program with
  | Some expression -> Pixel_loop expression
  | None -> Region_loop program
