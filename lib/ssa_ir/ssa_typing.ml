(* The typing rule of every opcode, in one place: the builder asks it for the
   result types it mints, the verifier asks it for the types a hand-built
   instruction must declare. The value results only; an effectful operation
   also returns an effect, appended by the caller. *)

(* The ordinal of an operand in {!Ssa_op.operands} order. *)
module Position =
  Core.Tagged_int.Make
    (struct
      let prefix = "operand"
    end)
    ()

type error =
  | Operand of {
      position : Position.t;
      expected : Ssa_type.t;
      found : Ssa_type.t;
    }
  | Operand_not_float of { position : Position.t; found : Ssa_type.t }
  | Operands_differ of { first : Ssa_type.t; second : Ssa_type.t }
  | Scale_out_of_domain of int64

let pp_error fmt = function
  | Operand { position; expected; found } ->
      Fmt.pf fmt "%a: expected %a, found %a" Position.pp position Ssa_type.pp
        expected Ssa_type.pp found
  | Operand_not_float { position; found } ->
      Fmt.pf fmt "%a: expected a float, found %a" Position.pp position
        Ssa_type.pp found
  | Operands_differ { first; second } ->
      Fmt.pf fmt "operand types differ: %a and %a" Ssa_type.pp first Ssa_type.pp
        second
  | Scale_out_of_domain k ->
      Fmt.pf fmt "scale literal %Ld is outside the index domain" k

let scalar s = Ssa_type.Scalar s
let index = scalar Ssa_type.Index
let f64 = scalar Ssa_type.F64

let expect position expected (v : Ssa_value.t) =
  if Ssa_type.equal v.Ssa_value.ty expected then Ok ()
  else Error (Operand { position; expected; found = v.Ssa_value.ty })

let ( let* ) = Result.bind

(* Operands from [first], each required to be [expected]. *)
let expect_all ~first expected vs =
  let rec go n = function
    | [] -> Ok ()
    | v :: rest ->
        let* () = expect (Position.of_int n) expected v in
        go (n + 1) rest
  in
  go first vs

let result_types : Ssa_op.t -> (Ssa_type.t list, error) result = function
  | Ssa_op.Const c -> Ok [ Ssa_const.ty c ]
  | Ssa_op.Convert (c, a) ->
      let from, into =
        match c with
        | Ssa_op.Convert.F32_to_f64 -> (scalar Ssa_type.F32, f64)
        | Ssa_op.Convert.F64_to_f32 -> (f64, scalar Ssa_type.F32)
        | Ssa_op.Convert.Index_to_f64 -> (index, f64)
        | Ssa_op.Convert.Index_to_i64 -> (index, scalar Ssa_type.I64)
      in
      let* () = expect (Position.of_int 0) from a in
      Ok [ into ]
  | Ssa_op.Float_binary (_, a, b) -> (
      match a.Ssa_value.ty with
      | Ssa_type.Scalar (Ssa_type.F32 | Ssa_type.F64) ->
          if Ssa_type.equal a.Ssa_value.ty b.Ssa_value.ty then
            Ok [ a.Ssa_value.ty ]
          else
            Error
              (Operands_differ
                 { first = a.Ssa_value.ty; second = b.Ssa_value.ty })
      | found ->
          Error (Operand_not_float { position = Position.of_int 0; found }))
  | Ssa_op.Index_add (a, b) ->
      let* () = expect_all ~first:0 index [ a; b ] in
      Ok [ index ]
  | Ssa_op.Index_scale (k, a) ->
      if not (Ssa_const.in_index_domain k) then Error (Scale_out_of_domain k)
      else
        let* () = expect (Position.of_int 0) index a in
        Ok [ index ]
  | Ssa_op.Load { at; decode; _ } ->
      let* () = expect_all ~first:0 index (Ssa_access.operands at) in
      Ok [ Ssa_op.Decode.result decode ]
  | Ssa_op.Mark _ -> Ok []
  | Ssa_op.Store { at; encode; value; _ } ->
      let coords = Ssa_access.operands at in
      let* () = expect_all ~first:0 index coords in
      let* () =
        expect
          (Position.of_int (List.length coords))
          (Ssa_op.Encode.operand encode)
          value
      in
      Ok []
