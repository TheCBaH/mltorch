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
    in_h : int;
    in_w : int;
    kernel_h : int;
    kernel_w : int;
    stride_h : int;
    stride_w : int;
    pad_h : int;
    pad_w : int;
    out : Role.Position.t Index.t Coord.t;
    result : result;
  }
end

type t = Max_pool of Max_pool.t

type geometry_field =
  [ `In_h
  | `In_w
  | `Kernel_h
  | `Kernel_w
  | `Pad_h
  | `Pad_w
  | `Stride_h
  | `Stride_w ]

type geometry_bound = [ `Non_negative | `Positive ]

module Bad_geometry = struct
  type t = { field : geometry_field; value : int; bound : geometry_bound }
end

type error = [ `Bad_geometry of Bad_geometry.t | Checked.error ]

let geometry_field_name : geometry_field -> string = function
  | `In_h -> "in_h"
  | `In_w -> "in_w"
  | `Kernel_h -> "kernel_h"
  | `Kernel_w -> "kernel_w"
  | `Pad_h -> "pad_h"
  | `Pad_w -> "pad_w"
  | `Stride_h -> "stride_h"
  | `Stride_w -> "stride_w"

let pp_error fmt : [< error ] -> unit = function
  | `Bad_geometry { Bad_geometry.field; value; bound } ->
      Fmt.pf fmt "%s must be %s, got %d"
        (geometry_field_name field)
        (match bound with `Non_negative -> ">= 0" | `Positive -> "> 0")
        value
  | #Checked.error as e -> Checked.pp_error fmt e

let max_pool ~source ~in_h ~in_w ~kernel_h ~kernel_w ~stride_h ~stride_w ~pad_h
    ~pad_w ~out ~result =
  let bounded bound ok field n =
    if ok n then Err.return n
    else Err.fail (`Bad_geometry { Bad_geometry.field; value = n; bound })
  in
  let positive = bounded `Positive (fun n -> n >= 1) in
  let nonneg = bounded `Non_negative (fun n -> n >= 0) in
  let open Err.Syntax in
  let* in_h = positive `In_h in_h in
  let* in_w = positive `In_w in_w in
  let* kernel_h = positive `Kernel_h kernel_h in
  let* kernel_w = positive `Kernel_w kernel_w in
  let* stride_h = positive `Stride_h stride_h in
  let* stride_w = positive `Stride_w stride_w in
  let* pad_h = nonneg `Pad_h pad_h in
  let+ pad_w = nonneg `Pad_w pad_w in
  Max_pool
    {
      Max_pool.source;
      in_h;
      in_w;
      kernel_h;
      kernel_w;
      stride_h;
      stride_w;
      pad_h;
      pad_w;
      out;
      result;
    }

let window (Max_pool d) ~out_h ~out_w =
  let open Err.Syntax in
  let axis out stride pad kernel extent =
    let* base = Checked.mul out stride in
    let* base = Checked.sub base pad in
    let+ top = Checked.add base kernel in
    (Stdlib.max 0 base, Stdlib.min extent top)
  in
  let* hlo, hhi =
    axis out_h d.Max_pool.stride_h d.Max_pool.pad_h d.Max_pool.kernel_h
      d.Max_pool.in_h
  in
  let+ wlo, whi =
    axis out_w d.Max_pool.stride_w d.Max_pool.pad_w d.Max_pool.kernel_w
      d.Max_pool.in_w
  in
  { Window.hlo; hhi; wlo; whi }

let flat_index (Max_pool d) ~ih ~iw =
  let open Err.Syntax in
  let* row = Checked.mul ih d.Max_pool.in_w in
  Checked.add row iw
