(* Concrete internally; the compatibility signature on [Expr.Intrinsic]
   retains the public private constructors. *)

module Window = struct
  type t = { hlo : int; hhi : int; wlo : int; whi : int }
end

module Max_pool = struct
  type result = Index | Value

  let result_name = function Index -> "index" | Value -> "value"

  type t = {
    source : Source.t;
    input : Core.Dim.extent Core.Dim.t Core.Geometry.Hw.t;
    kernel : Core.Dim.extent Core.Dim.t Core.Geometry.Hw.t;
    stride : Core.Geometry.Pos.t Core.Geometry.Hw.t;
    pad : Core.Geometry.Nonneg.t Core.Geometry.Hw.t;
    out : Role.Position.t Index.t Coord.t;
    result : result;
  }

  (* The eight geometry fields as plain ints, in the order [input], [kernel],
     [stride], [pad] (h before w): the one exit, for ordering, hashing and
     printing. *)
  let geometry d =
    let open Core.Geometry.Hw in
    [
      (d.input.h :> int);
      (d.input.w :> int);
      (d.kernel.h :> int);
      (d.kernel.w :> int);
      (d.stride.h :> int);
      (d.stride.w :> int);
      (d.pad.h :> int);
      (d.pad.w :> int);
    ]
end

type t = Max_pool of Max_pool.t
type error = Checked.error

let pp_error = Checked.pp_error

let max_pool ~source ~input ~kernel ~stride ~pad ~out ~result =
  Max_pool { Max_pool.source; input; kernel; stride; pad; out; result }

let window (Max_pool d) ~out_h ~out_w =
  let open Err.Syntax in
  let axis out stride pad kernel extent =
    let* base = Checked.mul out stride in
    let* base = Checked.sub base pad in
    let+ top = Checked.add base kernel in
    (Stdlib.max 0 base, Stdlib.min extent top)
  in
  let* hlo, hhi =
    axis out_h
      (d.Max_pool.stride.h :> int)
      (d.Max_pool.pad.h :> int)
      (d.Max_pool.kernel.h :> int)
      (d.Max_pool.input.h :> int)
  in
  let+ wlo, whi =
    axis out_w
      (d.Max_pool.stride.w :> int)
      (d.Max_pool.pad.w :> int)
      (d.Max_pool.kernel.w :> int)
      (d.Max_pool.input.w :> int)
  in
  { Window.hlo; hhi; wlo; whi }

let flat_index (Max_pool d) ~ih ~iw =
  let open Err.Syntax in
  let* row = Checked.mul ih (d.Max_pool.input.w :> int) in
  Checked.add row iw
