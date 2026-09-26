(* See dim_arith.mli. A [Pos.t] is at least 1, so re-entering [Dim] as an
   extent through its checked constructor cannot fail. *)

module Delta = struct
  let scale ~by d = Dim.Delta.scale (by : Op_config.Pos.t :> int) d

  let floor_div_pos d ~by =
    Dim.Delta.floor_div_pos d ~by:(Dim.extent (by : Op_config.Pos.t :> int))

  let ceil_div_pos d ~by =
    Dim.Delta.ceil_div_pos d ~by:(Dim.extent (by : Op_config.Pos.t :> int))
end

module Extent = struct
  let scale ~limit ~by e =
    Dim.product_bounded ~limit [ e; Dim.extent (by : Op_config.Pos.t :> int) ]
end
