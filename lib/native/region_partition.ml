module Axis_mode = struct
  type t = Singleton | Whole

  let pp fmt = function
    | Singleton -> Fmt.string fmt "singleton"
    | Whole -> Fmt.string fmt "whole"
end

type t = Axis_mode.t Vec6.t

type error =
  [ `Duplicate_axis of Expr.Axis.t | `Output_out_of_bounds of Vec6.coord ]

let pp_error fmt : [< error ] -> unit = function
  | `Duplicate_axis axis ->
      Fmt.pf fmt "duplicate whole axis %a" Expr.Axis.pp axis
  | `Output_out_of_bounds coord ->
      Fmt.pf fmt "output coordinate %a is out of bounds" Vec6.pp_coord coord

let singleton = Vec6.of_fn (fun _ -> Axis_mode.Singleton)

let of_whole_axes axes =
  let rec collect seen = function
    | [] ->
        Err.return
          (Vec6.mapi
             (fun axis _ ->
               if List.mem axis seen then Axis_mode.Whole
               else Axis_mode.Singleton)
             singleton)
    | axis :: rest ->
        if List.mem axis seen then Err.fail (`Duplicate_axis axis)
        else collect (axis :: seen) rest
  in
  collect [] axes

let mode partition axis = Vec6.get partition axis

let is_singleton partition =
  List.for_all (fun axis -> mode partition axis = Axis_mode.Singleton) Axis.all

let whole_axes partition =
  List.filter (fun axis -> mode partition axis = Axis_mode.Whole) Axis.all

let key_shape ~output_shape partition =
  Vec6.mapi
    (fun axis extent ->
      match mode partition axis with
      | Axis_mode.Singleton -> extent
      | Axis_mode.Whole -> Dim.extent 1)
    output_shape

let key_of_output ~output_shape partition output =
  if not (Vec6.in_bounds output_shape output) then
    Err.fail (`Output_out_of_bounds output)
  else
    Err.return
      (Vec6.mapi
         (fun axis value ->
           match mode partition axis with
           | Axis_mode.Singleton -> value
           | Axis_mode.Whole -> Dim.index 0)
         output)

let fold_keys ~output_shape ~init ~f partition =
  Vec6.fold_coords (key_shape ~output_shape partition) ~init ~f

let fold_outputs ~output_shape ~key ~init ~f partition =
  (* A key comes from [fold_keys], whose shape fixes whole axes at zero and
     bounds singleton axes by the output shape.  Enumerate only the axes owned
     by the key: scanning [output_shape] here once per key made materializing
     a partition with R keys cost R passes over the whole domain. *)
  let owned_shape =
    Vec6.mapi
      (fun axis extent ->
        match mode partition axis with
        | Axis_mode.Singleton -> Dim.extent 1
        | Axis_mode.Whole -> extent)
      output_shape
  in
  Vec6.fold_coords owned_shape ~init ~f:(fun acc varying ->
      let output =
        Vec6.mapi
          (fun axis _ ->
            match mode partition axis with
            | Axis_mode.Singleton -> Vec6.get key axis
            | Axis_mode.Whole -> Vec6.get varying axis)
          output_shape
      in
      f acc output)

let pp fmt partition =
  Fmt.list ~sep:(Fmt.any " ")
    (fun fmt axis ->
      Fmt.pf fmt "%a=%a" Expr.Axis.pp axis Axis_mode.pp (mode partition axis))
    fmt Axis.all
