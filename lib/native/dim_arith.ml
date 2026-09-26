(* See dim_arith.mli. A [Pos.t] is at least 1, so re-entering [Dim] as an
   extent through its checked constructor cannot fail. *)

let extent_of_pos (p : Op_config.Pos.t) = Dim.extent (p :> int)

module Delta = struct
  let scale ~by d = Dim.Delta.scale (by : Op_config.Pos.t :> int) d
  let floor_div_pos d ~by = Dim.Delta.floor_div_pos d ~by:(extent_of_pos by)
  let ceil_div_pos d ~by = Dim.Delta.ceil_div_pos d ~by:(extent_of_pos by)
end

module Extent = struct
  let of_pos = extent_of_pos
  let to_pos (e : Dim.extent Dim.t) = Op_config.Pos.of_int (e :> int)
  let div_exact ~by e = Dim.div_exact e ~by:(extent_of_pos by)
  let scale ~limit ~by e = Dim.product_bounded ~limit [ e; extent_of_pos by ]
end
