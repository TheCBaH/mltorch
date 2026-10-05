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
  | Divisor_not_positive of int64
  | I64_vector
  | Lane_out_of_range of { lane : Ssa_type.Lane.t; lanes : Ssa_type.Lanes.t }
  | Lanes_differ of { first : Ssa_type.Lanes.t; second : Ssa_type.Lanes.t }
  | Lanes_out_of_range of Ssa_type.Lanes.t
  | Not_lanewise of Ssa_op.t
  | Operand of {
      position : Position.t;
      expected : Ssa_type.t;
      found : Ssa_type.t;
    }
  | Operand_not_float of { position : Position.t; found : Ssa_type.t }
  | Operand_not_numeric of { position : Position.t; found : Ssa_type.t }
  | Operand_not_vector of { position : Position.t; found : Ssa_type.t }
  | Operands_differ of { first : Ssa_type.t; second : Ssa_type.t }
  | Scale_out_of_domain of int64

let pp_error fmt = function
  | Divisor_not_positive k ->
      Fmt.pf fmt "divisor %Ld is not a positive index literal" k
  | I64_vector -> Fmt.string fmt "a vector holds floats and masks, never int64"
  | Lane_out_of_range { lane; lanes } ->
      Fmt.pf fmt "%a is outside a vector of %a lanes" Ssa_type.Lane.pp lane
        Ssa_type.Lanes.pp lanes
  | Lanes_differ { first; second } ->
      Fmt.pf fmt "lane counts differ: %a and %a" Ssa_type.Lanes.pp first
        Ssa_type.Lanes.pp second
  | Lanes_out_of_range l ->
      Fmt.pf fmt "%a lanes is outside 1..%d" Ssa_type.Lanes.pp l
        Ssa_type.max_lanes
  | Not_lanewise op ->
      Fmt.pf fmt "%s cannot be applied lane by lane" (Ssa_op.name op)
  | Operand_not_vector { position; found } ->
      Fmt.pf fmt "%a: expected a vector or mask, found %a" Position.pp position
        Ssa_type.pp found
  | Operand { position; expected; found } ->
      Fmt.pf fmt "%a: expected %a, found %a" Position.pp position Ssa_type.pp
        expected Ssa_type.pp found
  | Operand_not_float { position; found } ->
      Fmt.pf fmt "%a: expected a float, found %a" Position.pp position
        Ssa_type.pp found
  | Operand_not_numeric { position; found } ->
      Fmt.pf fmt "%a: expected a number, found %a" Position.pp position
        Ssa_type.pp found
  | Operands_differ { first; second } ->
      Fmt.pf fmt "operand types differ: %a and %a" Ssa_type.pp first Ssa_type.pp
        second
  | Scale_out_of_domain k ->
      Fmt.pf fmt "scale literal %Ld is outside the index domain" k

let scalar s = Ssa_type.Scalar s
let index = scalar Ssa_type.Index
let f64 = scalar Ssa_type.F64
let i64 = scalar Ssa_type.I64

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

let pred = scalar Ssa_type.Pred

(* Both operands the same float type, which is also the result's. *)
let same_float (a : Ssa_value.t) (b : Ssa_value.t) =
  match a.Ssa_value.ty with
  | Ssa_type.Scalar (Ssa_type.F32 | Ssa_type.F64) ->
      if Ssa_type.equal a.Ssa_value.ty b.Ssa_value.ty then Ok a.Ssa_value.ty
      else
        Error
          (Operands_differ { first = a.Ssa_value.ty; second = b.Ssa_value.ty })
  | found -> Error (Operand_not_float { position = Position.of_int 0; found })

let steps_in_domain steps =
  match
    List.find_opt
      (fun k -> not (Ssa_const.in_index_domain k))
      (Expr.Coord.to_list steps)
  with
  | Some k -> Error (Scale_out_of_domain k)
  | None -> Ok ()

let positive_divisor k =
  if Int64.compare k 0L > 0 && Ssa_const.in_index_domain k then Ok ()
  else Error (Divisor_not_positive k)

let lanes_valid l =
  let n = Ssa_type.Lanes.to_int l in
  if n >= 1 && n <= Ssa_type.max_lanes then Ok ()
  else Error (Lanes_out_of_range l)

(* The vector type of a scalar element, if a vector can hold it. *)
let vector_of lanes = function
  | Ssa_type.Scalar ((Ssa_type.F32 | Ssa_type.F64) as s) ->
      Ok (Ssa_type.Vec (s, lanes))
  | Ssa_type.Scalar Ssa_type.Pred -> Ok (Ssa_type.Mask lanes)
  | Ssa_type.Scalar (Ssa_type.I64 | Ssa_type.Index | Ssa_type.Offset)
  | Ssa_type.Effect | Ssa_type.Local | Ssa_type.Mask _ | Ssa_type.Vec _ ->
      Error I64_vector

(* The lane count and scalar element of a vector or mask operand. *)
let element_of position (v : Ssa_value.t) =
  match v.Ssa_value.ty with
  | Ssa_type.Vec (s, l) -> Ok (l, Ssa_type.Scalar s)
  | Ssa_type.Mask l -> Ok (l, Ssa_type.Scalar Ssa_type.Pred)
  | found -> Error (Operand_not_vector { position; found })

let lanewise_allowed : Ssa_op.t -> bool = function
  | Ssa_op.Convert ((Ssa_op.Convert.F32_to_f64 | Ssa_op.Convert.F64_to_f32), _)
  | Ssa_op.Float_binary _ | Ssa_op.Float_compare _ | Ssa_op.Float_fma _
  | Ssa_op.Float_max _ | Ssa_op.Float_unary _ | Ssa_op.Pool_better _
  | Ssa_op.Pred_not _ | Ssa_op.Pred_or _ | Ssa_op.Select _ ->
      true
  | _ -> false

let rec result_types : Ssa_op.t -> (Ssa_type.t list, error) result = function
  | Ssa_op.Lanewise inner ->
      if not (lanewise_allowed inner) then Error (Not_lanewise inner)
      else
        (* the scalar operation on one lane, typed by its own rule *)
        let operands = Ssa_op.operands inner in
        let rec views n acc lanes = function
          | [] -> Ok (List.rev acc, lanes)
          | (v : Ssa_value.t) :: rest -> (
              let* l, elt = element_of (Position.of_int n) v in
              match lanes with
              | Some first when not (Ssa_type.Lanes.equal first l) ->
                  Error (Lanes_differ { first; second = l })
              | Some _ | None ->
                  views (n + 1)
                    (({ v with Ssa_value.ty = elt } : Ssa_value.t) :: acc)
                    (Some l) rest)
        in
        let* scalars, lanes = views 0 [] None operands in
        let lanes =
          match lanes with
          | Some l -> l
          | None ->
              invalid_arg "Ssa_typing: a lane-wise operation without operands"
        in
        let* () = lanes_valid lanes in
        let table = Hashtbl.create 4 in
        List.iter2
          (fun (v : Ssa_value.t) (s : Ssa_value.t) ->
            Hashtbl.replace table (v.Ssa_value.id :> int) s)
          operands scalars;
        let scalar_op =
          Ssa_op.map_operands
            (fun (v : Ssa_value.t) ->
              Hashtbl.find table (v.Ssa_value.id :> int))
            inner
        in
        let* results = result_types scalar_op in
        let rec lift acc = function
          | [] -> Ok (List.rev acc)
          | t :: rest ->
              let* v = vector_of lanes t in
              lift (v :: acc) rest
        in
        lift [] results
  | Ssa_op.Mark_lanes { lanes; _ } ->
      let* () = lanes_valid lanes in
      Ok []
  | Ssa_op.Vec_extract { lane; vector } ->
      let* lanes, elt = element_of (Position.of_int 0) vector in
      if
        Ssa_type.Lane.to_int lane < 0
        || Ssa_type.Lane.to_int lane >= Ssa_type.Lanes.to_int lanes
      then Error (Lane_out_of_range { lane; lanes })
      else Ok [ elt ]
  | Ssa_op.Vec_insert { lane; vector; element } ->
      let* lanes, elt = element_of (Position.of_int 0) vector in
      if
        Ssa_type.Lane.to_int lane < 0
        || Ssa_type.Lane.to_int lane >= Ssa_type.Lanes.to_int lanes
      then Error (Lane_out_of_range { lane; lanes })
      else
        let* () = expect (Position.of_int 1) elt element in
        Ok [ vector.Ssa_value.ty ]
  | Ssa_op.Vec_iota { base; step; lanes } ->
      let* () = lanes_valid lanes in
      if not (Ssa_const.in_index_domain step) then
        Error (Scale_out_of_domain step)
      else
        let* () = expect (Position.of_int 0) index base in
        Ok [ Ssa_type.Vec (Ssa_type.F64, lanes) ]
  | Ssa_op.Vec_splat { element; lanes } ->
      let* () = lanes_valid lanes in
      let* t = vector_of lanes element.Ssa_value.ty in
      Ok [ t ]
  | Ssa_op.Vec_load { at; steps; decode; lanes; _ } -> (
      let* () = lanes_valid lanes in
      let* () = expect_all ~first:0 index (Expr.Coord.to_list at) in
      let* () = steps_in_domain steps in
      match decode with
      | Ssa_op.Decode.I64 -> Error I64_vector
      | Ssa_op.Decode.Bf16_to_f64 | Ssa_op.Decode.Bool_to_f64
      | Ssa_op.Decode.F16_to_f64 | Ssa_op.Decode.F32_to_f64
      | Ssa_op.Decode.F64_to_f64 | Ssa_op.Decode.I16_dequant
      | Ssa_op.Decode.I32_to_f64 | Ssa_op.Decode.I64_to_f64
      | Ssa_op.Decode.I8_dequant ->
          Ok [ Ssa_type.Vec (Ssa_type.F64, lanes) ])
  | Ssa_op.Vec_store { at; steps; encode; value; lanes; _ } -> (
      let* () = lanes_valid lanes in
      let* () = expect_all ~first:0 index (Expr.Coord.to_list at) in
      let* () = steps_in_domain steps in
      match encode with
      | Ssa_op.Encode.I64 -> Error I64_vector
      | Ssa_op.Encode.Bool_nonzero | Ssa_op.Encode.F32_round ->
          let* () =
            expect (Position.of_int 6)
              (Ssa_type.Vec (Ssa_type.F64, lanes))
              value
          in
          Ok [])
  | Ssa_op.Check_access { at; _ } ->
      let* () = expect_all ~first:0 index (Ssa_access.operands at) in
      Ok []
  | Ssa_op.Check_local { at; extent; _ } ->
      let* () = positive_divisor extent in
      let* () = expect (Position.of_int 0) index at in
      Ok []
  | Ssa_op.Check_scan { row; lane; row_extent; lane_extent; _ } ->
      let* () = positive_divisor row_extent in
      let* () = positive_divisor lane_extent in
      let* () = expect_all ~first:0 index [ row; lane ] in
      Ok []
  | Ssa_op.Local_alloc { slots; _ } ->
      let* () = positive_divisor slots in
      Ok [ Ssa_type.Local ]
  | Ssa_op.Local_read { local; at } ->
      let* () = expect (Position.of_int 0) Ssa_type.Local local in
      let* () = expect (Position.of_int 1) index at in
      Ok [ f64 ]
  | Ssa_op.Local_write { local; at; value } ->
      let* () = expect (Position.of_int 0) Ssa_type.Local local in
      let* () = expect (Position.of_int 1) index at in
      let* () = expect (Position.of_int 2) f64 value in
      Ok []
  | Ssa_op.Meter_charge | Ssa_op.Meter_reset -> Ok []
  | Ssa_op.Meter_release width | Ssa_op.Meter_reserve width ->
      let* () = positive_divisor width in
      Ok []
  | Ssa_op.Float_compare (_, a, b) ->
      let* _ = same_float a b in
      Ok [ pred ]
  | Ssa_op.Check_gather { raw; extent } ->
      let* () = positive_divisor extent in
      let* () = expect (Position.of_int 0) i64 raw in
      Ok []
  | Ssa_op.Float_max (a, b) ->
      let* t = same_float a b in
      Ok [ t ]
  | Ssa_op.Float_fma (a, b, c) ->
      let* t = same_float a b in
      let* _ = same_float b c in
      Ok [ t ]
  | Ssa_op.Float_to_i64 a ->
      let* () = expect (Position.of_int 0) f64 a in
      Ok [ i64 ]
  | Ssa_op.I64_arith (_, a, b) | Ssa_op.I64_div (a, b) ->
      let* () = expect_all ~first:0 i64 [ a; b ] in
      Ok [ i64 ]
  | Ssa_op.I64_compare (_, a, b) ->
      let* () = expect_all ~first:0 i64 [ a; b ] in
      Ok [ pred ]
  | Ssa_op.Index_of_i64 a ->
      let* () = expect (Position.of_int 0) i64 a in
      Ok [ index ]
  | Ssa_op.Float_unary (_, a) ->
      let* t = same_float a a in
      Ok [ t ]
  | Ssa_op.Index_ceil_div (k, a) | Ssa_op.Index_floor_div (k, a) ->
      let* () = positive_divisor k in
      let* () = expect (Position.of_int 0) index a in
      Ok [ index ]
  | Ssa_op.Index_clamp_low a ->
      let* () = expect (Position.of_int 0) index a in
      Ok [ index ]
  | Ssa_op.Index_compare (_, a, b) ->
      let* () = expect_all ~first:0 index [ a; b ] in
      Ok [ pred ]
  | Ssa_op.Index_max (a, b) | Ssa_op.Index_min (a, b) ->
      let* () = expect_all ~first:0 index [ a; b ] in
      Ok [ index ]
  | Ssa_op.Pool_better (a, b) ->
      let* _ = same_float a b in
      Ok [ pred ]
  | Ssa_op.Pred_not a ->
      let* () = expect (Position.of_int 0) pred a in
      Ok [ pred ]
  | Ssa_op.Pred_or (a, b) ->
      let* () = expect_all ~first:0 pred [ a; b ] in
      Ok [ pred ]
  | Ssa_op.Select (p, a, b) -> (
      let* () = expect (Position.of_int 0) pred p in
      match a.Ssa_value.ty with
      | Ssa_type.Scalar
          (Ssa_type.F32 | Ssa_type.F64 | Ssa_type.I64 | Ssa_type.Index) ->
          if Ssa_type.equal a.Ssa_value.ty b.Ssa_value.ty then
            Ok [ a.Ssa_value.ty ]
          else
            Error
              (Operands_differ
                 { first = a.Ssa_value.ty; second = b.Ssa_value.ty })
      | found ->
          Error (Operand_not_numeric { position = Position.of_int 1; found }))
  | Ssa_op.Const c -> Ok [ Ssa_const.ty c ]
  | Ssa_op.Convert (c, a) ->
      let from, into =
        match c with
        | Ssa_op.Convert.F32_to_f64 -> (scalar Ssa_type.F32, f64)
        | Ssa_op.Convert.F64_to_f32 -> (f64, scalar Ssa_type.F32)
        | Ssa_op.Convert.I64_to_f32 -> (i64, scalar Ssa_type.F32)
        | Ssa_op.Convert.I64_to_f64 -> (i64, f64)
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
  | Ssa_op.Index_add (a, b) | Ssa_op.Index_add_in_domain (a, b) ->
      let* () = expect_all ~first:0 index [ a; b ] in
      Ok [ index ]
  | Ssa_op.Index_scale (k, a) | Ssa_op.Index_scale_in_domain (k, a) ->
      if not (Ssa_const.in_index_domain k) then Error (Scale_out_of_domain k)
      else
        let* () = expect (Position.of_int 0) index a in
        Ok [ index ]
  | Ssa_op.Load { at; decode; _ } | Ssa_op.Load_in_bounds { at; decode; _ } ->
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
