(* See alloc_script.mli. Sizes are [int64], bounded per edge by
   [Kernel.Limits.Hard.numel], so one edge's bytes cannot overflow; their sum is
   a different aggregate and is checked. *)

open Graph_common

module Kind = struct
  type t =
    | Float32
    | Float64
    | Int16_signed
    | Int16_unsigned
    | Int32
    | Int64
    | Int8_signed
    | Int8_unsigned

  let all =
    [
      Float32;
      Float64;
      Int16_signed;
      Int16_unsigned;
      Int32;
      Int64;
      Int8_signed;
      Int8_unsigned;
    ]

  let of_fmt (Payload.Fmt f) =
    match f with
    | Payload.BF16 | Payload.F16 -> Int16_unsigned
    | Payload.Bool -> Int8_unsigned
    | Payload.F32 -> Float32
    | Payload.F64 -> Float64
    | Payload.I16 -> Int16_signed
    | Payload.I32 -> Int32
    | Payload.I64 -> Int64
    | Payload.I8 -> Int8_signed

  let cell_bytes = function
    | Float32 | Int32 -> 4L
    | Float64 | Int64 -> 8L
    | Int16_signed | Int16_unsigned -> 2L
    | Int8_signed | Int8_unsigned -> 1L

  let equal (a : t) b = a = b
  let compare (a : t) b = Stdlib.compare a b

  let name = function
    | Float32 -> "float32"
    | Float64 -> "float64"
    | Int16_signed -> "int16_signed"
    | Int16_unsigned -> "int16_unsigned"
    | Int32 -> "int32"
    | Int64 -> "int64"
    | Int8_signed -> "int8_signed"
    | Int8_unsigned -> "int8_unsigned"

  let pp ppf t = Format.pp_print_string ppf (name t)
end

module Alloc = struct
  type t = {
    id : Tensor_id.t;
    signature : Tensor_sig.t;
    kind : Kind.t;
    numel : int64;
    bytes : int64;
    eligible : bool;
  }

  let fmt_name (Payload.Fmt f) = Payload.fmt_name f

  let equal a b =
    let sa = a.signature and sb = b.signature in
    Tensor_id.equal a.id b.id && Kind.equal a.kind b.kind && a.numel = b.numel
    && a.bytes = b.bytes && a.eligible = b.eligible
    && sa.Tensor_sig.shape = sb.Tensor_sig.shape
    && String.equal (fmt_name sa.Tensor_sig.fmt) (fmt_name sb.Tensor_sig.fmt)
    && Option.equal Quant.equal sa.Tensor_sig.quant sb.Tensor_sig.quant
end

module Event = struct
  type t = Alloc of Alloc.t | Free of Tensor_id.t | Node of Node_id.t

  let equal a b =
    match (a, b) with
    | Alloc a, Alloc b -> Alloc.equal a b
    | Free a, Free b -> Tensor_id.equal a b
    | Node a, Node b -> Node_id.equal a b
    | (Alloc _ | Free _ | Node _), _ -> false

  let pp ppf = function
    | Alloc a ->
        Format.fprintf ppf "alloc %a %a %Ld bytes%s" Tensor_id.pp a.Alloc.id
          Kind.pp a.Alloc.kind a.Alloc.bytes
          (if a.Alloc.eligible then "" else " (outside the arena)")
    | Free id -> Format.fprintf ppf "free %a" Tensor_id.pp id
    | Node id -> Format.fprintf ppf "node %a" Node_id.pp id
end

type t = Event.t list

let alloc ~released (sg : Tensor_sig.t) =
  let open Err.Syntax in
  let+ numel =
    Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel sg.Tensor_sig.shape
  in
  let bytes =
    Int64.mul numel (Int64.of_int (Payload.packed_cell_bytes sg.Tensor_sig.fmt))
  in
  {
    Alloc.id = sg.Tensor_sig.id;
    signature = sg;
    kind = Kind.of_fmt sg.Tensor_sig.fmt;
    numel;
    bytes;
    eligible =
      Tensor_id.Set.mem sg.Tensor_sig.id released
      && Option.is_none sg.Tensor_sig.quant;
  }

module Position =
  Core.Tagged_int.Make
    (struct
      let prefix = "@"
    end)
    ()

module Difference = struct
  type t = {
    position : Position.t;
    left : Event.t option;
    right : Event.t option;
  }
end

let first_difference (a : t) (b : t) =
  let rec go i a b =
    match (a, b) with
    | [], [] -> None
    | x :: a', y :: b' when Event.equal x y -> go (i + 1) a' b'
    | _ ->
        let head = function [] -> None | x :: _ -> Some x in
        Some
          {
            Difference.position = Position.of_int i;
            left = head a;
            right = head b;
          }
  in
  go 0 a b

let add id acc bytes =
  if Int64.compare acc (Int64.sub Int64.max_int bytes) > 0 then
    Err.fail (`Peak_bytes_overflow id)
  else Err.return (Int64.add acc bytes)

let peak_bytes (script : t) =
  let open Err.Syntax in
  let+ _, _, peak =
    Err.List.fold_left
      (fun (resident, live, peak) event ->
        match event with
        | Event.Node _ -> Err.return (resident, live, peak)
        | Event.Alloc a ->
            let+ resident = add a.Alloc.id resident a.Alloc.bytes in
            ( resident,
              Tensor_id.Map.add a.Alloc.id a.Alloc.bytes live,
              Int64.max peak resident )
        | Event.Free id -> (
            match Tensor_id.Map.find_opt id live with
            | None -> Err.return (resident, live, peak)
            | Some bytes ->
                Err.return
                  (Int64.sub resident bytes, Tensor_id.Map.remove id live, peak)
            ))
      (0L, Tensor_id.Map.empty, 0L)
      script
  in
  peak

let out_of_arena_bytes (script : t) =
  Err.List.fold_left
    (fun acc -> function
      | Event.Alloc a when not a.Alloc.eligible ->
          add a.Alloc.id acc a.Alloc.bytes
      | Event.Alloc _ | Event.Free _ | Event.Node _ -> Err.return acc)
    0L script

let pp ppf script =
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list ~pp_sep:Format.pp_print_cut Event.pp)
    script
