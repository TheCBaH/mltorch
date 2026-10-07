(* A language failure: its kind with the static identity the kind owns, and the
   runtime payload a [Fail] terminator supplies as values. These are exactly the
   SSA interpreter's failure rows; a compiler invariant or defect is never one.

   The record a selected program stores is the existing model failure record:
   [kind : int32] at byte 0, [invocation : int32] at byte 4, then
   [record_words] [int64] words from byte 8. Each kind keeps its own word
   schema. Only [Scan_projection] (word 5) and [Unbound_local] (word 0) hold a
   site word, an index into the bundle's own failure-site table; no other kind
   has a site field. *)

module Overflow_op = struct
  type t = Add | Mul

  let name = function Add -> "add" | Mul -> "mul"
end

module Meter = struct
  type t = State_over_limit | Updates_exhausted

  let name = function
    | State_over_limit -> "state_over_limit"
    | Updates_exhausted -> "updates_exhausted"
end

module Scan_axis = struct
  type t = Lane | Row

  let name = function Lane -> "lane" | Row -> "row"
end

module Scan = struct
  type t = { which : Scan_axis.t; var : Expr.Local_var.t option }
end

module Coord = struct
  type t = { source : Expr.Source.t; axis : Expr.Axis.t }
end

type t =
  | Coord_out_of_range of Coord.t
  | Gather_index_out_of_range
  | I64_division_by_zero
  | I64_division_overflow
  | I64_from_float_infinite
  | I64_from_float_nan
  | I64_from_float_out_of_range
  | Index_overflow of Overflow_op.t
  | Scan_meter of Meter.t
  | Scan_projection of Scan.t
  | Unbound_local of Expr.Local_var.t

let name = function
  | Coord_out_of_range _ -> "coord_out_of_range"
  | Gather_index_out_of_range -> "gather_index_out_of_range"
  | I64_division_by_zero -> "i64_division_by_zero"
  | I64_division_overflow -> "i64_division_overflow"
  | I64_from_float_infinite -> "i64_from_float_infinite"
  | I64_from_float_nan -> "i64_from_float_nan"
  | I64_from_float_out_of_range -> "i64_from_float_out_of_range"
  | Index_overflow _ -> "index_overflow"
  | Scan_meter _ -> "scan_meter"
  | Scan_projection _ -> "scan_projection"
  | Unbound_local _ -> "unbound_local"

let equal_var a b =
  match (a, b) with
  | Some a, Some b -> Expr.Local_var.equal a b
  | None, None -> true
  | Some _, None | None, Some _ -> false

(* Static identity: kind and every static field. *)
let equal a b =
  match (a, b) with
  | Coord_out_of_range a, Coord_out_of_range b ->
      Expr.Source.equal a.Coord.source b.Coord.source
      && Expr.Axis.equal a.Coord.axis b.Coord.axis
  | Index_overflow a, Index_overflow b -> a = b
  | Scan_meter a, Scan_meter b -> a = b
  | Scan_projection a, Scan_projection b ->
      a.Scan.which = b.Scan.which && equal_var a.Scan.var b.Scan.var
  | Unbound_local a, Unbound_local b -> Expr.Local_var.equal a b
  | ( ( Gather_index_out_of_range | I64_division_by_zero | I64_division_overflow
      | I64_from_float_infinite | I64_from_float_nan
      | I64_from_float_out_of_range ),
      _ ) ->
      name a = name b
  | ( ( Coord_out_of_range _ | Index_overflow _ | Scan_meter _
      | Scan_projection _ | Unbound_local _ ),
      _ ) ->
      false

(* The types of the payload a [Fail] supplies, in order. A coordinate is its six
   components in axis order; every integer field is an [i64] holding the value
   sign-extended from its source domain. *)
let payload_types = function
  | Coord_out_of_range _ -> List.init 6 (fun _ -> Mir_type.i64)
  | Gather_index_out_of_range -> [ Mir_type.i64; Mir_type.i64 ]
  | I64_division_by_zero | I64_division_overflow | I64_from_float_infinite
  | I64_from_float_nan | Unbound_local _ ->
      []
  | I64_from_float_out_of_range -> [ Mir_type.F64 ]
  | Index_overflow _ -> [ Mir_type.i64; Mir_type.i64 ]
  | Scan_meter _ -> [ Mir_type.i64 ]
  | Scan_projection _ -> [ Mir_type.i64; Mir_type.i64; Mir_type.i64 ]

(* Whether the kind's record holds a site word. *)
let payload = payload_types

let uses_site = function
  | Scan_projection _ | Unbound_local _ -> true
  | Coord_out_of_range _ | Gather_index_out_of_range | I64_division_by_zero
  | I64_division_overflow | I64_from_float_infinite | I64_from_float_nan
  | I64_from_float_out_of_range | Index_overflow _ | Scan_meter _ ->
      false

(* The record's kind word: the position in the established kind enumeration,
   whose slot 1 is the decoder's defect kind. The numbers are an ABI. *)
let kind_word = function
  | Coord_out_of_range _ -> 0l
  | Gather_index_out_of_range -> 2l
  | I64_division_by_zero -> 3l
  | I64_division_overflow -> 4l
  | I64_from_float_infinite -> 5l
  | I64_from_float_nan -> 6l
  | I64_from_float_out_of_range -> 7l
  | Index_overflow _ -> 8l
  | Scan_meter _ -> 9l
  | Scan_projection _ -> 10l
  | Unbound_local _ -> 11l

let record_words = 12
let record_bytes = 104L
let record_kind_offset = 0L
let record_invocation_offset = 4L
let record_word_offset k = Int64.add 8L (Int64.mul 8L (Int64.of_int k))

(* An entry of a bundle's failure-site table, as far as a record decodes it:
   the kind and local variable, never the expressions a source check holds. *)
module Site_entry = struct
  type t =
    | Local_out_of_range of Expr.Local_var.t
    | Other  (** a table entry no site word of these kinds can name *)
    | Scan_lane_out_of_range of Expr.Local_var.t option
    | Scan_row_out_of_range of Expr.Local_var.t option

  let compatible entry failure =
    match (entry, failure) with
    | Local_out_of_range v, Unbound_local w -> Expr.Local_var.equal v w
    | Scan_lane_out_of_range v, Scan_projection { Scan.which = Lane; var } ->
        equal_var v var
    | Scan_row_out_of_range v, Scan_projection { Scan.which = Row; var } ->
        equal_var v var
    | ( ( Local_out_of_range _ | Other | Scan_lane_out_of_range _
        | Scan_row_out_of_range _ ),
        _ ) ->
        false
end

(* The site word of a site-bearing failure: the first compatible entry of the
   supplied table, as the SSA C emitter selects it. [None] for a reachable
   failure with no compatible entry — a refusal, never a sentinel. *)
let bind_site ~(table : Site_entry.t array) failure =
  let n = Array.length table in
  let rec go i =
    if i >= n then None
    else if Site_entry.compatible table.(i) failure then
      Some (Mir_id.Site.of_int i)
    else go (i + 1)
  in
  if uses_site failure then go 0 else None

let axis_word a = Int64.of_int (Expr.Axis.to_int a)

(* Where each record word comes from: a constant of the static identity, a
   payload value (by position; a float payload as its bits), or the site.
   Every word not listed is zero. [words] and selection's record stores both
   read this, so the two cannot drift. *)
module Word = struct
  type t = Const of int64 | Payload of int | Site
end

let layout = function
  | Coord_out_of_range { Coord.source; axis } ->
      [
        (0, Word.Const (Int64.of_int (Expr.Source.to_int source)));
        (1, Word.Const (axis_word axis));
        (2, Word.Payload (Expr.Axis.to_int axis));
      ]
      @ List.init 6 (fun k -> (3 + k, Word.Payload k))
  | Gather_index_out_of_range -> [ (0, Word.Payload 0); (1, Word.Payload 1) ]
  | I64_from_float_out_of_range -> [ (0, Word.Payload 0) ]
  | Index_overflow op ->
      [
        ( 0,
          Word.Const
            (match op with Overflow_op.Add -> 0L | Overflow_op.Mul -> 1L) );
        (1, Word.Payload 0);
        (2, Word.Payload 1);
      ]
  | Scan_meter m ->
      [
        ( 0,
          Word.Const
            (match m with
            | Meter.State_over_limit -> 0L
            | Meter.Updates_exhausted -> 1L) );
        (1, Word.Payload 0);
      ]
  | Scan_projection { Scan.which; var } ->
      [
        ( 0,
          Word.Const
            (match which with Scan_axis.Lane -> 0L | Scan_axis.Row -> 1L) );
        (1, Word.Const (if Option.is_some var then 1L else 0L));
        (2, Word.Payload 0);
        (3, Word.Payload 1);
        (4, Word.Payload 2);
        (5, Word.Site);
      ]
  | Unbound_local _ -> [ (0, Word.Site) ]
  | I64_division_by_zero | I64_division_overflow | I64_from_float_infinite
  | I64_from_float_nan ->
      []

(* The words of a record, from the static identity, the payload (each an int64:
   a float payload as its bits) and the site a site-bearing kind needs. *)
let words failure ~(payload : int64 list) ~(site : Mir_id.Site.t option) =
  if List.length payload <> List.length (payload_types failure) then
    invalid_arg "Mir_failure.words: payload does not match the kind";
  let v = Array.make record_words 0L in
  List.iter
    (fun (k, w) ->
      v.(k) <-
        (match w with
        | Word.Const c -> c
        | Word.Payload i -> List.nth payload i
        | Word.Site -> (
            match site with
            | Some s -> Int64.of_int (Mir_id.Site.to_int s)
            | None ->
                invalid_arg
                  "Mir_failure.words: a site-bearing kind needs a site")))
    (layout failure);
  v

let pp_var fmt = function
  | Some v -> Expr.Local_var.pp fmt v
  | None -> Fmt.string fmt "inline"

let pp fmt = function
  | Coord_out_of_range { Coord.source; axis } ->
      Fmt.pf fmt "coord_out_of_range(%a, %a)" Expr.Source.pp source Expr.Axis.pp
        axis
  | Index_overflow op -> Fmt.pf fmt "index_overflow(%s)" (Overflow_op.name op)
  | Scan_meter m -> Fmt.pf fmt "scan_meter(%s)" (Meter.name m)
  | Scan_projection { Scan.which; var } ->
      Fmt.pf fmt "scan_projection(%s, %a)" (Scan_axis.name which) pp_var var
  | Unbound_local v -> Fmt.pf fmt "unbound_local(%a)" Expr.Local_var.pp v
  | ( Gather_index_out_of_range | I64_division_by_zero | I64_division_overflow
    | I64_from_float_infinite | I64_from_float_nan | I64_from_float_out_of_range
      ) as f ->
      Fmt.string fmt (name f)

module Decode_error = struct
  type t =
    | Malformed  (** a word out of its field's range *)
    | Sentinel_site
        (** the one-past-table site: an invariant case was reached — a compiler
            defect, never a language failure *)
    | Site_disagrees  (** the site's entry is not this kind's *)
    | Unknown_kind of int32

  let pp fmt = function
    | Malformed -> Fmt.string fmt "malformed failure record"
    | Sentinel_site -> Fmt.string fmt "the record names the sentinel site"
    | Site_disagrees -> Fmt.string fmt "the record disagrees with its site"
    | Unknown_kind k -> Fmt.pf fmt "unknown failure kind %ld" k
end

let axis_of_word w =
  List.find_opt (fun a -> Int64.equal (axis_word a) w) Expr.Axis.all

(* A record back to its static identity, payload words and raw site: the
   inverse of [words], resolving a site through the bundle's table. *)
let decode ~(table : Site_entry.t array) ~kind ~(v : int64 array) =
  let n = Array.length table in
  let site i =
    let w = v.(i) in
    if Int64.equal w (Int64.of_int n) then Error Decode_error.Sentinel_site
    else if Int64.compare w 0L < 0 || Int64.compare w (Int64.of_int n) > 0 then
      Error Decode_error.Malformed
    else Ok (Int64.to_int w)
  in
  let source_of w =
    if Int64.compare w 0L >= 0 && Int64.compare w 0x7FFF_FFFFL <= 0 then
      Some (Expr.Source.create (Int64.to_int w))
    else None
  in
  if Array.length v <> record_words then Error Decode_error.Malformed
  else
    match kind with
    | 0l -> (
        match (source_of v.(0), axis_of_word v.(1)) with
        | Some source, Some axis ->
            Ok
              ( Coord_out_of_range { Coord.source; axis },
                Array.to_list (Array.sub v 3 6),
                None )
        | _ -> Error Decode_error.Malformed)
    | 2l -> Ok (Gather_index_out_of_range, [ v.(0); v.(1) ], None)
    | 3l -> Ok (I64_division_by_zero, [], None)
    | 4l -> Ok (I64_division_overflow, [], None)
    | 5l -> Ok (I64_from_float_infinite, [], None)
    | 6l -> Ok (I64_from_float_nan, [], None)
    | 7l -> Ok (I64_from_float_out_of_range, [ v.(0) ], None)
    | 8l -> (
        match v.(0) with
        | 0L -> Ok (Index_overflow Overflow_op.Add, [ v.(1); v.(2) ], None)
        | 1L -> Ok (Index_overflow Overflow_op.Mul, [ v.(1); v.(2) ], None)
        | _ -> Error Decode_error.Malformed)
    | 9l -> (
        match v.(0) with
        | 0L -> Ok (Scan_meter Meter.State_over_limit, [ v.(1) ], None)
        | 1L -> Ok (Scan_meter Meter.Updates_exhausted, [ v.(1) ], None)
        | _ -> Error Decode_error.Malformed)
    | 10l -> (
        match site 5 with
        | Error e -> Error e
        | Ok s -> (
            let payload = [ v.(2); v.(3); v.(4) ] in
            let site = Some (Mir_id.Site.of_int s) in
            match (v.(0), table.(s)) with
            | 0L, Site_entry.Scan_lane_out_of_range var ->
                Ok
                  ( Scan_projection { Scan.which = Scan_axis.Lane; var },
                    payload,
                    site )
            | 1L, Site_entry.Scan_row_out_of_range var ->
                Ok
                  ( Scan_projection { Scan.which = Scan_axis.Row; var },
                    payload,
                    site )
            | _ -> Error Decode_error.Site_disagrees))
    | 11l -> (
        match site 0 with
        | Error e -> Error e
        | Ok s -> (
            match table.(s) with
            | Site_entry.Local_out_of_range var ->
                Ok (Unbound_local var, [], Some (Mir_id.Site.of_int s))
            | _ -> Error Decode_error.Site_disagrees))
    | k -> Error (Decode_error.Unknown_kind k)
