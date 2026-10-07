(* The result types of a generic operation, checked against its operand types
   and immediates. The one place an opcode's typing rule lives: the verifier
   and the builder both ask it. *)

module Immediate = struct
  (* An immediate outside the domain its opcode admits. Closed. *)
  type t =
    | Alignment  (** not a power of two up to the access width *)
    | Bitcast_pair  (** not i32<->f32 or i64<->f64 *)
    | Const_bits  (** noncanonical bits, or a type a constant cannot have *)
    | Event_count  (** below one *)
    | Width_order
        (** an extension that does not widen, a truncation that does not narrow
        *)

  let name = function
    | Alignment -> "alignment"
    | Bitcast_pair -> "bitcast_pair"
    | Const_bits -> "const_bits"
    | Event_count -> "event_count"
    | Width_order -> "width_order"
end

module Error = struct
  type t =
    | Arity of { expected : int; found : int }
    | Bad_immediate of Immediate.t
    | Bad_operand of { position : int; found : Mir_type.t }
    | Unknown_callee of Mir_op.Callee.t
    | Unknown_view of Mir_id.View.t

  let pp fmt = function
    | Arity { expected; found } ->
        Fmt.pf fmt "%d operands where the signature takes %d" found expected
    | Bad_immediate i -> Fmt.pf fmt "bad immediate: %s" (Immediate.name i)
    | Bad_operand { position; found } ->
        Fmt.pf fmt "operand %d has type %a" position Mir_type.pp found
    | Unknown_callee c -> Fmt.pf fmt "unknown callee %a" Mir_op.Callee.pp c
    | Unknown_view v -> Fmt.pf fmt "unknown view %a" Mir_id.View.pp v
end

(* A callee's parameter and result types. *)
module Signature = struct
  type t = { params : Mir_type.t list; results : Mir_type.t list }
end

let ( let* ) = Result.bind

let operand position (v : Mir_value.t) ok =
  if ok v.Mir_value.ty then Ok ()
  else Error (Error.Bad_operand { position; found = v.Mir_value.ty })

let same position (v : Mir_value.t) ty = operand position v (Mir_type.equal ty)
let float_ty = Mir_type.is_float
let int_ty = Mir_type.is_int
let width_of = function Mir_type.Int w -> Some w | _ -> None
let immediate ok i = if ok then Ok () else Error (Error.Bad_immediate i)

let check ~(signature : Mir_op.Callee.t -> Signature.t option)
    ~(view : Mir_id.View.t -> bool) (op : Mir_op.t) =
  let open Mir_op in
  match op with
  | Addr v ->
      if view v then Ok [ Mir_type.Ptr ] else Error (Error.Unknown_view v)
  | Bitcast (ty, a) ->
      let pair =
        match (a.Mir_value.ty, ty) with
        | Mir_type.Int Mir_width.W32, Mir_type.F32
        | Mir_type.F32, Mir_type.Int Mir_width.W32
        | Mir_type.Int Mir_width.W64, Mir_type.F64
        | Mir_type.F64, Mir_type.Int Mir_width.W64 ->
            true
        | _ -> false
      in
      let* () = immediate pair Immediate.Bitcast_pair in
      Ok [ ty ]
  | Call (c, args) -> (
      match signature c with
      | None -> Error (Error.Unknown_callee c)
      | Some { Signature.params; results } ->
          let n = List.length params and m = List.length args in
          if n <> m then Error (Error.Arity { expected = n; found = m })
          else
            let rec go i ps args =
              match (ps, args) with
              | p :: ps, a :: args ->
                  let* () = same i a p in
                  go (i + 1) ps args
              | _ -> Ok results
            in
            go 0 params args)
  | Const c ->
      let* () = immediate (Mir_const.well_formed c) Immediate.Const_bits in
      Ok [ c.Mir_const.ty ]
  | Copy a ->
      let* () = operand 0 a Mir_type.has_storage in
      Ok [ a.Mir_value.ty ]
  | Event (_, n) ->
      let* () = immediate (Int64.compare n 1L >= 0) Immediate.Event_count in
      Ok []
  | Fbinary (_, a, b) ->
      let* () = operand 0 a float_ty in
      let* () = same 1 b a.Mir_value.ty in
      Ok [ a.Mir_value.ty ]
  | Fcmp (_, a, b) ->
      let* () = operand 0 a float_ty in
      let* () = same 1 b a.Mir_value.ty in
      Ok [ Mir_type.Pred ]
  | Fconvert (c, a) ->
      let src, dst =
        match c with
        | Fconvert.F32_to_f64 -> (Mir_type.F32, Mir_type.F64)
        | Fconvert.F64_to_f32 -> (Mir_type.F64, Mir_type.F32)
        | Fconvert.S64_to_f32 -> (Mir_type.i64, Mir_type.F32)
        | Fconvert.S64_to_f64 -> (Mir_type.i64, Mir_type.F64)
      in
      let* () = same 0 a src in
      Ok [ dst ]
  | Ffma (a, b, c) ->
      let* () = operand 0 a float_ty in
      let* () = same 1 b a.Mir_value.ty in
      let* () = same 2 c a.Mir_value.ty in
      Ok [ a.Mir_value.ty ]
  | Fto_sint a ->
      let* () = same 0 a Mir_type.F64 in
      Ok [ Mir_type.i64 ]
  | Funary (_, a) ->
      let* () = operand 0 a float_ty in
      Ok [ a.Mir_value.ty ]
  | Iarith (_, a, b) | Idiv (_, a, b) ->
      let* () = operand 0 a int_ty in
      let* () = same 1 b a.Mir_value.ty in
      Ok [ a.Mir_value.ty ]
  | Icmp (_, a, b) ->
      let* () = operand 0 a int_ty in
      let* () = same 1 b a.Mir_value.ty in
      Ok [ Mir_type.Pred ]
  | Iext (_, w, a) -> (
      let* () = operand 0 a int_ty in
      match width_of a.Mir_value.ty with
      | Some w0 when Mir_width.bits w0 < Mir_width.bits w ->
          Ok [ Mir_type.Int w ]
      | _ -> Error (Error.Bad_immediate Immediate.Width_order))
  | Itrunc (w, a) | Narrow (w, a) -> (
      let* () = operand 0 a int_ty in
      match width_of a.Mir_value.ty with
      | Some w0 when Mir_width.bits w0 > Mir_width.bits w ->
          Ok [ Mir_type.Int w ]
      | _ -> Error (Error.Bad_immediate Immediate.Width_order))
  | Load { Access.width; addr; align } ->
      let* () = same 0 addr Mir_type.Ptr in
      let* () =
        immediate
          (Mir_layout.is_power_of_two align
          && Int64.compare align (Mir_width.bytes width) <= 0)
          Immediate.Alignment
      in
      Ok [ Mir_type.Int width ]
  | Pbinary (_, a, b) ->
      let* () = same 0 a Mir_type.Pred in
      let* () = same 1 b Mir_type.Pred in
      Ok [ Mir_type.Pred ]
  | Pnot a ->
      let* () = same 0 a Mir_type.Pred in
      Ok [ Mir_type.Pred ]
  | Ptr_add (a, b) ->
      let* () = same 0 a Mir_type.Ptr in
      let* () = same 1 b Mir_type.i64 in
      Ok [ Mir_type.Ptr ]
  | Select (p, a, b) ->
      let* () = same 0 p Mir_type.Pred in
      let* () =
        operand 1 a (function
          | Mir_type.Order | Mir_type.Mask _ | Mir_type.Vec _ -> false
          | _ -> true)
      in
      let* () = same 2 b a.Mir_value.ty in
      Ok [ a.Mir_value.ty ]
  | Store ({ Access.width; addr; align }, v) ->
      let* () = same 0 addr Mir_type.Ptr in
      let* () = same 1 v (Mir_type.Int width) in
      let* () =
        immediate
          (Mir_layout.is_power_of_two align
          && Int64.compare align (Mir_width.bytes width) <= 0)
          Immediate.Alignment
      in
      Ok []
