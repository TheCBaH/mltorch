(* A tensor is just a 6D shape plus a typed payload — dense, channels-last, no
   strides (native_tensor_design.md §1c). The element at a coord is
   [Vec6.in_bounds] (strict) -> [Vec6.offset] -> [Payload.get_float]. *)

type ('elt, 'ba, 'q) t = {
  shape : Vec6.shape;
  payload : ('elt, 'ba, 'q) Payload.payload;
}

(* The concrete format is hidden; tensors flow through the graph as [packed]. *)
type packed = Tensor : ('elt, 'ba, 'q) t -> packed

let channel c = Dim.to_int (Vec6.get c Axis.C)

(* Strict read of one element as a float: every axis index must lie within the
   source extent, else [Invalid_argument]. This is the one read primitive — it
   does NOT broadcast or pad. A caller that needs an extent-1 axis to fan out
   reduces the coordinate first ([Pointwise.broadcast_coord]); one that needs a pad value
   outside the source takes a guarded tap ([shift_in_bounds]) and supplies its
   own. See .ai/native_tensor_design.md §1b. *)
let read (Tensor t) (coord : Vec6.coord) =
  if not (Vec6.in_bounds t.shape coord) then
    invalid_arg
      (Format.asprintf "Tensor.read: coord %a out of bounds for shape %a"
         Vec6.pp_coord coord Vec6.pp_shape t.shape);
  let i = (Vec6.offset t.shape coord :> int) in
  Payload.get_float t.payload ~c:(channel coord) ~i

(* Strict read at 6 explicit indices, one per axis — the primitive ops reach
   through the semantics' [load6], which makes this the innermost
   per-element path for every real op (millions of calls per real
   conv/pool/etc). Explicit args, not a closure: a closure built fresh per
   call (as [conv.ml]'s [x_idx]/[w_idx] used to be) allocates, and profiling
   found that dominated remaining allocation even after the [Vec6.coord]
   fixes below — see .ai/pt2_inference_perf.md. Args are already-validated
   [Dim.index Dim.t] (every producer of a [position index] in [Direct] goes
   through [Dim.index]/[Dim.succ] — see direct.ml), so [Vec6.offset_of] (the
   allocation-free, no-[coord]-needed fused check+offset built for exactly
   this call site) needs no re-validation, only the upper bound; the rare
   out-of-range case falls back to building the coord and calling [read], so
   its [Invalid_argument] behavior (message included) is unchanged. *)
let read_at6 (Tensor tr as packed) ~(n : Dim.index Dim.t) ~(t : Dim.index Dim.t)
    ~(d : Dim.index Dim.t) ~(h : Dim.index Dim.t) ~(w : Dim.index Dim.t)
    ~(c : Dim.index Dim.t) =
  let i = Vec6.offset_of tr.shape ~n ~t ~d ~h ~w ~c in
  if i >= 0 then Payload.get_float tr.payload ~c:(c :> int) ~i
  else
    read packed
      (Vec6.coord
         ~n:(n :> int)
         ~t:(t :> int)
         ~d:(d :> int)
         ~h:(h :> int)
         ~w:(w :> int)
         ~c:(c :> int))

(* Closure-based counterpart, for callers that don't (yet) go through
   [load6]'s 6 explicit args — see .ai/pt2_inference_perf.md for which ops
   still use this. *)
let read_at packed (idx : Axis.t -> Dim.index Dim.t) =
  read_at6 packed ~n:(idx N) ~t:(idx T) ~d:(idx D) ~h:(idx H) ~w:(idx W)
    ~c:(idx C)

(* Same as [read_at], for a caller whose index function is naturally raw
   [int] instead of [Dim.index Dim.t] — [Expr.Eval.value]'s grounding of a [Load]
   node (Symbolic-expression interpretation, not [Direct]): [Expr]'s own
   arithmetic is untyped int, unrelated to [Dim.t]/[Direct]'s representation,
   so there's no pre-validated [Dim.index Dim.t] to hand [read_at] here. Not
   the hot path [read_at] is optimized for. *)
let read_at_raw packed (idx : Axis.t -> int) =
  read packed
    (Vec6.coord ~n:(idx N) ~t:(idx T) ~d:(idx D) ~h:(idx H) ~w:(idx W)
       ~c:(idx C))

(* tap helper: the source coord = [base] + per-axis signed [deltas], guarded into
   the source extents. [None] is the pad region. *)
let shift_in_bounds (Tensor t) (base : Vec6.coord) (deltas : Vec6.deltas) =
  let idx a = Dim.to_int (Vec6.get base a) + Dim.to_int (Vec6.get deltas a) in
  let in_range a = idx a >= 0 && idx a < Dim.to_int (Vec6.get t.shape a) in
  if List.for_all in_range Axis.all then
    Some
      (Vec6.coord ~n:(idx N) ~t:(idx T) ~d:(idx D) ~h:(idx H) ~w:(idx W)
         ~c:(idx C))
  else None

(* Build a fresh dense float32 tensor by running [f] over every output coord. *)
let materialize (shape : Vec6.shape) (f : Vec6.coord -> float) =
  let n = (Vec6.numel shape :> int) in
  let data = Bigarray.(Array1.create float32 c_layout n) in
  let payload = { Payload.fmt = Payload.F32; quant = Payload.No_quant; data } in
  Vec6.iter shape (fun c ->
      let i = (Vec6.offset shape c :> int) in
      Payload.set_float payload ~c:(channel c) ~i (f c));
  Tensor { shape; payload }

(* [Unbind] is a storage-preserving selection, unlike the arithmetic ops whose
   results enter the engine's f32 compute domain.  Copy cells directly so an
   i64 value (including one beyond float's exact range) is never routed through
   [Payload.get_float].  Keeping this primitive here also gives Native and
   Native4D one implementation of the dtype-preserving path. *)
let unbind (Tensor src) ~axis ~output ~shape =
  let source_coord out =
    let zero = Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:0 in
    let base =
      List.fold_left
        (fun v (input_axis, output_axis) ->
          Vec6.copy out ~src:output_axis ~dst:input_axis v)
        zero
        (Aten_shape.repack_dropped ~dropped:[ axis ])
    in
    Vec6.set base axis (Dim.index output)
  in
  let copy : type e b q. (e, b, q) Payload.payload -> packed =
   fun payload ->
    let copy_data data =
      Vec6.iter shape (fun out ->
          let dst = (Vec6.offset shape out :> int) in
          let src = (Vec6.offset src.shape (source_coord out) :> int) in
          data.{dst} <- payload.data.{src})
    in
    let n = (Vec6.numel shape :> int) in
    match payload.fmt with
    | Payload.F32 ->
        let data =
          Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout n
        in
        copy_data data;
        Tensor { shape; payload = { payload with data } }
    | Payload.I64 ->
        let data = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout n in
        copy_data data;
        Tensor { shape; payload = { payload with data } }
    | Payload.I32 ->
        let data = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
        copy_data data;
        Tensor { shape; payload = { payload with data } }
    | Payload.F16 ->
        let data =
          Bigarray.Array1.create Bigarray.int16_unsigned Bigarray.c_layout n
        in
        copy_data data;
        Tensor { shape; payload = { payload with data } }
    | Payload.BF16 ->
        let data =
          Bigarray.Array1.create Bigarray.int16_unsigned Bigarray.c_layout n
        in
        copy_data data;
        Tensor { shape; payload = { payload with data } }
    | Payload.I16 ->
        let data =
          Bigarray.Array1.create Bigarray.int16_signed Bigarray.c_layout n
        in
        copy_data data;
        Tensor { shape; payload = { payload with data } }
    | Payload.I8 ->
        let data =
          Bigarray.Array1.create Bigarray.int8_signed Bigarray.c_layout n
        in
        copy_data data;
        Tensor { shape; payload = { payload with data } }
  in
  copy src.payload

(* Bit-for-bit equality — the only correct notion when the engine claims two
   computations are *identical* rather than merely close.

   [Float.equal] is unfit for that and was silently in use across the tests:
   [Float.equal (-0.) 0.] is [true], and so is [Float.equal] on two NaNs with
   different payloads. Both are exactly the distinctions the max-pool semantics
   turn on (see [Max_op]), so a test built on [Float.equal] cannot observe the
   bug it is meant to pin.

   Compares [Int64] bits of the *decoded* element rather than [Int32] bits of the
   stored one: decoding is what every consumer sees, the widening is exact for
   every storage format (so no case analysis on [Payload.fmt] is needed), and it
   still separates signed zeros and distinct NaN payloads. *)
let equal_bits (Tensor a as ta) (Tensor b as tb) =
  let bits x = Int64.bits_of_float x in
  Stdlib.( = ) a.shape b.shape
  && Vec6.fold_coords a.shape ~init:true ~f:(fun ok coord ->
      ok && Int64.equal (bits (read ta coord)) (bits (read tb coord)))

(* ---- JSON codec ----------------------------------------------------------- *)

(* Encodes a packed tensor as
     {"fmt":"f32","quant":...,"shape":[...],"data":{"Array":[...]} or {"None":null}}
   [max_elts]: if Some n and the tensor has more than n elements, the data is
   encoded as {"None":null} instead of {"Array":[...]}.  Decoders always accept
   both forms; {"None":null} decodes to [None].  I64 cells use JSON strings so
   their complete signed 64-bit value survives JavaScript and round-trips. *)
let jsont ?(max_elts : int option) () : packed Jsont.t =
  let enc_packed (Tensor t) =
    let numel = (Vec6.numel t.shape :> int) in
    let data_json =
      match t.payload.Payload.fmt with
      | Payload.I64 ->
          if Option.fold ~none:false ~some:(fun m -> numel > m) max_elts then
            Json_util.single ~case:"None" Json_util.jnull
          else
            Json_util.single ~case:"Array"
              (Json_util.jarr
                 (List.init numel (fun i ->
                      Json_util.enc Jsont.int64_as_string t.payload.data.{i})))
      | _ ->
          Payload.enc_data_union ~max_elts ~numel ~iter_floats:(fun yield ->
              Vec6.iter t.shape (fun c ->
                  let i = (Vec6.offset t.shape c :> int) in
                  yield i (Payload.get_float t.payload ~c:(channel c) ~i)))
    in
    let quant_json =
      match t.payload.Payload.quant with
      | Payload.No_quant -> Json_util.jnull
      | Payload.Quant q -> Json_util.enc Quant.jsont q
    in
    Json_util.jobj
      [
        ("data", data_json);
        ( "fmt",
          Json_util.enc Payload.packed_fmt_jsont
            (Payload.Fmt t.payload.Payload.fmt) );
        ("quant", quant_json);
        ("shape", Json_util.enc Vec6.shape_jsont t.shape);
      ]
  in
  let dec_packed json =
    let ms = Json_util.req_obj json "tensor" in
    let get k c = Json_util.req_field ms k c "tensor" in
    let shape = get "shape" Vec6.shape_jsont in
    let fmt = get "fmt" Payload.packed_fmt_jsont in
    let data_json =
      match Json_util.find_member ms "data" with
      | Some v -> v
      | None -> Jsont.Error.msgf Jsont.Meta.none "tensor: missing \"data\""
    in
    match data_json with
    | Jsont.Object ([ (("None", _), _) ], _) ->
        Jsont.Error.msgf Jsont.Meta.none
          "tensor: cannot decode {\"None\":null} — data was elided"
    | Jsont.Object ([ (("Array", _), Jsont.Array (vs, _)) ], _) -> (
        let expected = (Vec6.numel shape :> int) in
        let got = List.length vs in
        if got <> expected then
          Jsont.Error.msgf Jsont.Meta.none
            "tensor: data has %d elements but shape needs %d" got expected;
        match fmt with
        | Payload.Fmt Payload.F32 ->
            let data = Bigarray.(Array1.create float32 c_layout expected) in
            List.iteri
              (fun i v -> data.{i} <- Json_util.dec Json_util.f32_jsont v)
              vs;
            Tensor
              {
                shape;
                payload =
                  { Payload.fmt = Payload.F32; quant = Payload.No_quant; data };
              }
        | Payload.Fmt Payload.I64 ->
            let data = Bigarray.(Array1.create int64 c_layout expected) in
            List.iteri
              (fun i v -> data.{i} <- Json_util.dec Jsont.int64_as_string v)
              vs;
            Tensor
              {
                shape;
                payload =
                  { Payload.fmt = Payload.I64; quant = Payload.No_quant; data };
              }
        | Payload.Fmt fmt ->
            Jsont.Error.msgf Jsont.Meta.none
              "tensor: decoding format %s is not supported"
              (Payload.fmt_name fmt))
    | _ ->
        Jsont.Error.msgf Jsont.Meta.none
          "tensor: \"data\" must be {\"Array\":[...]} or {\"None\":null}"
  in
  Jsont.map ~kind:"tensor" ~dec:dec_packed ~enc:enc_packed Jsont.json

let pp fmt (Tensor t) =
  Format.fprintf fmt "tensor %a %a {" Payload.pp t.payload Vec6.pp_shape t.shape;
  let n = (Vec6.numel t.shape :> int) in
  let k = min n 8 in
  let count = ref 0 in
  (try
     Vec6.iter t.shape (fun c ->
         if !count >= k then raise Exit;
         if !count > 0 then Format.fprintf fmt ", ";
         let i = (Vec6.offset t.shape c :> int) in
         (match t.payload.Payload.fmt with
         | Payload.I64 -> Format.fprintf fmt "%Ld" t.payload.data.{i}
         | Payload.I32 -> Format.fprintf fmt "%ld" t.payload.data.{i}
         | _ ->
             Format.fprintf fmt "%g"
               (Payload.get_float t.payload ~c:(channel c) ~i));
         incr count)
   with Exit -> ());
  if n > k then Format.fprintf fmt ", ...";
  Format.fprintf fmt "}"
