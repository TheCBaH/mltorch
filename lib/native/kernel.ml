(* See kernel.mli. *)

module Binding = struct
  type t = Caller | Captured_constant | Filled of float

  let pp fmt = function
    | Caller -> Fmt.string fmt "caller"
    | Captured_constant -> Fmt.string fmt "constant"
    | Filled v -> Fmt.pf fmt "filled %g" v
end

module Input = struct
  type t = { id : Tensor_id.t; sg : Tensor_sig.t; binding : Binding.t }
end

module Result_conversion = struct
  type t = Round_f32

  let apply Round_f32 e = Expr.Value.round_f32 e
  let name Round_f32 = "round_f32"
end

module Value = struct
  type t = {
    id : Tensor_id.t;
    sg : Tensor_sig.t;
    body : Expr.Value.t;
    result : Result_conversion.t;
  }
end

module Output = struct
  type t = { value : Tensor_id.t; sg : Tensor_sig.t }
end

module Limits = struct
  type t = {
    max_size : int;
    max_depth : int;
    max_values : int;
    max_dep_depth : int;
    max_inputs : int;
    max_outputs : int;
    max_extent : int64;
    max_numel : int64;
  }

  module Invalid = struct
    type t = { name : string; value : int64 }
  end

  type error = [ `Invalid_limit of Invalid.t ]

  let pp_error fmt : [< error ] -> unit = function
    | `Invalid_limit { Invalid.name; value } ->
        Fmt.pf fmt "invalid limit %s = %Ld" name value

  module Hard = struct
    (* [depth] and [eval_depth] come from test/native/depth_probe.ml, measured
       under node: every traversal survives 1024 there and the first failures
       are at 2048 (Pp.value, Value.compare, Value.hash) — CHECK.VALUE STILL
       SURVIVES 2048, which is why the ceiling follows the minimum over all
       traversals rather than the checker's own figure. [Eval.value] is the
       outlier upward, surviving 4096, so the combined ceiling is higher; it has
       to be, since whole-program resnet18 reaches ~770 combined depth and the
       buffer-based evaluator never recurses through it. *)
    let depth = 256
    let eval_depth = 2048

    (* Memory and time, not stack. *)
    let size = 65536
    let values = 65536
    let dep_depth = 4096
    let inputs = 4096
    let outputs = 4096

    (* The JS-reachable runtime domain: extents, coordinates and reachable
       storage offsets stay below 2^31. *)
    let extent = 0x8000_0000L
    let numel = 0x8000_0000L
  end

  let check_int name v hard =
    if v <= 0 || v >= hard then
      Core.fail (`Invalid_limit { Invalid.name; value = Int64.of_int v })
    else Core.return ()

  let check_int64 name v hard =
    if Int64.compare v 0L <= 0 || Int64.compare v hard >= 0 then
      Core.fail (`Invalid_limit { Invalid.name; value = v })
    else Core.return ()

  let create ~max_size ~max_depth ~max_values ~max_dep_depth ~max_inputs
      ~max_outputs ~max_extent ~max_numel =
    let open Core.Syntax in
    let* () = check_int "max_size" max_size Hard.size in
    let* () = check_int "max_depth" max_depth Hard.depth in
    let* () = check_int "max_values" max_values Hard.values in
    let* () = check_int "max_dep_depth" max_dep_depth Hard.dep_depth in
    let* () = check_int "max_inputs" max_inputs Hard.inputs in
    let* () = check_int "max_outputs" max_outputs Hard.outputs in
    let* () = check_int64 "max_extent" max_extent Hard.extent in
    let+ () = check_int64 "max_numel" max_numel Hard.numel in
    {
      max_size;
      max_depth;
      max_values;
      max_dep_depth;
      max_inputs;
      max_outputs;
      max_extent;
      max_numel;
    }

  (* Census-derived (.ai/native_kernel_census.tsv): largest observed body size
     246, depth 14, 36 stages, dependency depth 6, 17 inputs / 16 outputs. Each
     default clears its maximum by an order of magnitude and sits at or below
     half the corresponding [Hard] ceiling. *)
  let default =
    Core.or_raise pp_error
      (create ~max_size:4096 ~max_depth:128 ~max_values:4096 ~max_dep_depth:1024
         ~max_inputs:1024 ~max_outputs:1024 ~max_extent:0x7FFF_FFFFL
         ~max_numel:0x7FFF_FFFFL)
end

type t = {
  inputs : Input.t list;
  values : Value.t list;
  outputs : Output.t list;
  limits : Limits.t;
}

module Sig_mismatch = struct
  type t = { record : Tensor_id.t; sg : Tensor_id.t }
end

module Unresolved = struct
  type t = { at : Tensor_id.t; source : Expr.Source.t }
end

module Forward_ref = struct
  type t = { at : Tensor_id.t; depends_on : Tensor_id.t }
end

module Extent_bound = struct
  type t = { id : Tensor_id.t; axis : Expr.Axis.t; extent : int64 }
end

module Format_rule = struct
  type role = Filled_input | Stored_value
  type t = { id : Tensor_id.t; role : role; fmt : Payload.packed_fmt }

  let role_name = function
    | Filled_input -> "filled input"
    | Stored_value -> "stored value"
end

module Body_error = struct
  type t = { at : Tensor_id.t; error : Expr.Check.error }
end

type error =
  [ `Duplicate_id of Tensor_id.t
  | `Signature_id_mismatch of Sig_mismatch.t
  | `Unresolved_source of Unresolved.t
  | `Forward_reference of Forward_ref.t
  | `Unknown_output of Tensor_id.t
  | `Unreachable_value of Tensor_id.t
  | `Too_many_values of int
  | `Too_many_inputs of int
  | `Too_many_outputs of int
  | `Dependency_too_deep of int
  | `Eval_too_deep of int
  | `Extent_too_large of Extent_bound.t
  | `Numel_too_large of Tensor_id.t
  | `Not_materializable of Format_rule.t
  | `Quant_contract of Tensor_id.t
  | `Body of Body_error.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Duplicate_id id -> Fmt.pf fmt "duplicate id %a" Tensor_id.pp id
  | `Signature_id_mismatch { Sig_mismatch.record; sg } ->
      Fmt.pf fmt "id %a disagrees with its signature id %a" Tensor_id.pp record
        Tensor_id.pp sg
  | `Unresolved_source { Unresolved.at; source } ->
      Fmt.pf fmt "%a loads unresolved source %a" Tensor_id.pp at Tensor_id.pp
        (Expr_bridge.id_of_source source)
  | `Forward_reference { Forward_ref.at; depends_on } ->
      Fmt.pf fmt "%a depends on later value %a" Tensor_id.pp at Tensor_id.pp
        depends_on
  | `Unknown_output id -> Fmt.pf fmt "output %a names no value" Tensor_id.pp id
  | `Unreachable_value id ->
      Fmt.pf fmt "value %a is not reachable from any output" Tensor_id.pp id
  | `Too_many_values n -> Fmt.pf fmt "more than %d values" n
  | `Too_many_inputs n -> Fmt.pf fmt "more than %d inputs" n
  | `Too_many_outputs n -> Fmt.pf fmt "more than %d outputs" n
  | `Dependency_too_deep n -> Fmt.pf fmt "dependency depth exceeds %d" n
  | `Eval_too_deep n -> Fmt.pf fmt "evaluation depth exceeds %d" n
  | `Extent_too_large { Extent_bound.id; axis; extent } ->
      Fmt.pf fmt "%a extent %a=%Ld exceeds the limit" Tensor_id.pp id
        Expr.Axis.pp axis extent
  | `Numel_too_large id ->
      Fmt.pf fmt "%a element count exceeds the limit" Tensor_id.pp id
  | `Not_materializable { Format_rule.id; role; fmt = f } ->
      Fmt.pf fmt "%a: a %s must be f32 and unquantized, got %s" Tensor_id.pp id
        (Format_rule.role_name role)
        (match f with Payload.Fmt g -> Payload.fmt_name g)
  | `Quant_contract id ->
      Fmt.pf fmt "%a: quantization disagrees with its format or channel extent"
        Tensor_id.pp id
  | `Body { Body_error.at; error } ->
      Fmt.pf fmt "%a: %a" Tensor_id.pp at Expr.Check.pp_error error

(* ---- bounds ---------------------------------------------------------------

   Checked BEFORE each operation, never on its result. Six extents each below
   2^31 have a mathematical product near 2^186, so folding them into an [int64]
   and comparing afterwards is a post-overflow comparison, not a bound. The
   guard divides instead: with every factor >= 1 and the accumulator proved <=
   limit, [acc * factor] cannot exceed the limit and so cannot wrap.

   Dense storage needs no separate stride-weighted sum today: [Vec6.offset] is
   the row-major linearisation, so the largest reachable offset is numel - 1 and
   bounding numel bounds it. A layout with explicit strides gets its own checked
   fold here. *)
module Bounds = struct
  let signature (limits : Limits.t) id (sg : Tensor_sig.t) =
    let open Core.Syntax in
    let* () =
      List.fold_left
        (fun acc axis ->
          let* () = acc in
          let e =
            Int64.of_int (Dim.to_int (Vec6.get sg.Tensor_sig.shape axis))
          in
          if Int64.compare e limits.Limits.max_extent > 0 then
            Core.fail (`Extent_too_large { Extent_bound.id; axis; extent = e })
          else Core.return ())
        (Core.return ()) Expr.Axis.all
    in
    let limit = limits.Limits.max_numel in
    let+ _ =
      List.fold_left
        (fun acc axis ->
          let* n = acc in
          let e =
            Int64.of_int (Dim.to_int (Vec6.get sg.Tensor_sig.shape axis))
          in
          if Int64.compare n (Int64.div limit e) > 0 then
            Core.fail (`Numel_too_large id)
          else Core.return (Int64.mul n e))
        (Core.return 1L) Expr.Axis.all
    in
    ()
end

(* ---- signature contracts -------------------------------------------------- *)

let quant_contract id (sg : Tensor_sig.t) =
  let quantized = Payload.is_quantized sg.Tensor_sig.fmt in
  match (quantized, sg.Tensor_sig.quant) with
  | false, None -> Core.return ()
  | true, Some q -> (
      (* A per-channel value's two arrays agree by [Quant.per_channel]'s
         construction; relating that common length to C is this side's job. *)
      match Quant.channel_count q with
      | None -> Core.return ()
      | Some n ->
          if n = Dim.to_int (Vec6.get sg.Tensor_sig.shape Expr.Axis.C) then
            Core.return ()
          else Core.fail (`Quant_contract id))
  | true, None | false, Some _ -> Core.fail (`Quant_contract id)

(* [Tensor.materialize] always produces an f32, unquantized payload, and it is
   the only materialiser the evaluator has. So a locally created tensor must
   declare what it will actually be handed; otherwise a kernel can advertise f16
   or quantized storage, pass every other check, and have [Expr_bridge.env]
   decode a real f32 payload while all analysis describes the declared format.
   Caller and captured inputs are unaffected — they carry real data and stay
   free to be f16/bf16/quantized. *)
let materializable id role (sg : Tensor_sig.t) =
  let f32 =
    match sg.Tensor_sig.fmt with Payload.Fmt Payload.F32 -> true | _ -> false
  in
  if f32 && Option.is_none sg.Tensor_sig.quant then Core.return ()
  else
    Core.fail
      (`Not_materializable { Format_rule.id; role; fmt = sg.Tensor_sig.fmt })

(* ---- construction --------------------------------------------------------- *)

let create ?(limits = Limits.default) ~inputs ~values ~outputs () =
  let open Core.Syntax in
  (* Arity first: the cheapest guards, and they bound the list and map work
     every later check performs. Neither list contributes to [max_values], so a
     kernel with a thousand unused inputs would otherwise satisfy every DAG
     budget while imposing unbounded setup on the evaluator, which validates
     each declared input before computing anything. *)
  let* () =
    if List.length inputs > limits.Limits.max_inputs then
      Core.fail (`Too_many_inputs limits.Limits.max_inputs)
    else Core.return ()
  in
  let* () =
    if List.length outputs > limits.Limits.max_outputs then
      Core.fail (`Too_many_outputs limits.Limits.max_outputs)
    else Core.return ()
  in
  let* () =
    if List.length values > limits.Limits.max_values then
      Core.fail (`Too_many_values limits.Limits.max_values)
    else Core.return ()
  in
  (* Budgets before any unmetered traversal. [Expr.Fold]'s queries walk the whole
     tree; running one first would exhaust the stack on precisely the oversized
     body the limit exists to reject. *)
  let* () =
    List.fold_left
      (fun acc (v : Value.t) ->
        let* () = acc in
        (* [Core.map_error], not a hand-rolled rebuild: it preserves the
           original detection backtrace, which unwrapping [.kind] and calling
           [Core.fail] silently would not. *)
        Core.map_error
          (fun e -> `Body { Body_error.at = v.id; error = e })
          (Expr.Check.value ~max_size:limits.Limits.max_size
             ~max_depth:limits.Limits.max_depth v.body))
      (Core.return ()) values
  in
  (* Identity: the record id and its signature id must agree, since [sg.id] is
     the binding key everywhere else. Leaving them independent would key source
     resolution and binding differently. *)
  let* () =
    let check id (sg : Tensor_sig.t) =
      if Tensor_id.equal id sg.Tensor_sig.id then Core.return ()
      else
        Core.fail
          (`Signature_id_mismatch
             { Sig_mismatch.record = id; sg = sg.Tensor_sig.id })
    in
    let* () =
      List.fold_left
        (fun acc (i : Input.t) ->
          let* () = acc in
          check i.id i.sg)
        (Core.return ()) inputs
    in
    List.fold_left
      (fun acc (v : Value.t) ->
        let* () = acc in
        check v.id v.sg)
      (Core.return ()) values
  in
  let* _seen =
    List.fold_left
      (fun acc id ->
        let* seen = acc in
        if Tensor_id.Set.mem id seen then Core.fail (`Duplicate_id id)
        else Core.return (Tensor_id.Set.add id seen))
      (Core.return Tensor_id.Set.empty)
      (List.map (fun (i : Input.t) -> i.Input.id) inputs
      @ List.map (fun (v : Value.t) -> v.Value.id) values)
  in
  let input_ids =
    List.fold_left
      (fun s (i : Input.t) -> Tensor_id.Set.add i.Input.id s)
      Tensor_id.Set.empty inputs
  in
  (* Source resolution, dependency depth and evaluation depth in one forward
     sweep over the already topologically ordered list — iterative, so
     validation cannot itself overflow on the input it exists to reject. Both
     depths are keyed by id; a source resolving to an input contributes zero. *)
  let* _depths =
    List.fold_left
      (fun acc (v : Value.t) ->
        let* dep, ev, defined = acc in
        let* d, e =
          Expr.Source.Set.fold
            (fun src acc ->
              let* d, e = acc in
              let id = Expr_bridge.id_of_source src in
              if Tensor_id.Set.mem id input_ids then Core.return (d, e)
              else
                match
                  (Tensor_id.Map.find_opt id dep, Tensor_id.Map.find_opt id ev)
                with
                | Some pd, Some pe -> Core.return (max d pd, max e pe)
                | _ ->
                    (* Defined later, or not at all: the ordered list makes
                       these the same walk. A source naming a value that exists
                       further down is a forward reference; anything else is
                       unresolved. *)
                    if Tensor_id.Set.mem id defined then
                      Core.fail
                        (`Forward_reference
                           { Forward_ref.at = v.Value.id; depends_on = id })
                    else
                      Core.fail
                        (`Unresolved_source
                           { Unresolved.at = v.Value.id; source = src }))
            (Expr.Fold.sources v.Value.body)
            (Core.return (0, 0))
        in
        let d = d + 1 and e = e + Expr.Fold.depth v.Value.body in
        let* () =
          if d > limits.Limits.max_dep_depth then
            Core.fail (`Dependency_too_deep limits.Limits.max_dep_depth)
          else Core.return ()
        in
        let* () =
          if e > Limits.Hard.eval_depth then
            Core.fail (`Eval_too_deep Limits.Hard.eval_depth)
          else Core.return ()
        in
        Core.return
          ( Tensor_id.Map.add v.Value.id d dep,
            Tensor_id.Map.add v.Value.id e ev,
            defined ))
      (Core.return
         ( Tensor_id.Map.empty,
           Tensor_id.Map.empty,
           List.fold_left
             (fun s (v : Value.t) -> Tensor_id.Set.add v.Value.id s)
             Tensor_id.Set.empty values ))
      values
  in
  (* Outputs name values; their signatures are derived, never supplied. *)
  let value_sig =
    List.fold_left
      (fun m (v : Value.t) -> Tensor_id.Map.add v.Value.id v.Value.sg m)
      Tensor_id.Map.empty values
  in
  let* out =
    List.fold_left
      (fun acc id ->
        let* out = acc in
        match Tensor_id.Map.find_opt id value_sig with
        | None -> Core.fail (`Unknown_output id)
        | Some sg -> Core.return ({ Output.value = id; sg } :: out))
      (Core.return []) outputs
  in
  let out = List.rev out in
  (* Reachability: a reverse sweep over the same ordered list, so no recursive
     graph walk. Everything a live value reads becomes live. *)
  let* () =
    let live =
      List.fold_left
        (fun s (o : Output.t) -> Tensor_id.Set.add o.Output.value s)
        Tensor_id.Set.empty out
    in
    let live =
      List.fold_left
        (fun live (v : Value.t) ->
          if Tensor_id.Set.mem v.Value.id live then
            Expr.Source.Set.fold
              (fun src s -> Tensor_id.Set.add (Expr_bridge.id_of_source src) s)
              (Expr.Fold.sources v.Value.body)
              live
          else live)
        live (List.rev values)
    in
    List.fold_left
      (fun acc (v : Value.t) ->
        let* () = acc in
        if Tensor_id.Set.mem v.Value.id live then Core.return ()
        else Core.fail (`Unreachable_value v.Value.id))
      (Core.return ()) values
  in
  (* Signature contracts and the runtime domain, for every declared tensor. *)
  let* () =
    List.fold_left
      (fun acc (i : Input.t) ->
        let* () = acc in
        let* () = quant_contract i.Input.id i.Input.sg in
        let* () =
          match i.Input.binding with
          | Binding.Filled _ ->
              materializable i.Input.id Format_rule.Filled_input i.Input.sg
          | Binding.Caller | Binding.Captured_constant -> Core.return ()
        in
        Bounds.signature limits i.Input.id i.Input.sg)
      (Core.return ()) inputs
  in
  let* () =
    List.fold_left
      (fun acc (v : Value.t) ->
        let* () = acc in
        let* () = quant_contract v.Value.id v.Value.sg in
        let* () =
          materializable v.Value.id Format_rule.Stored_value v.Value.sg
        in
        Bounds.signature limits v.Value.id v.Value.sg)
      (Core.return ()) values
  in
  Core.return { inputs; values; outputs = out; limits }

let pp fmt (k : t) =
  let comma l = String.concat ", " l in
  Fmt.pf fmt "@[<v>";
  List.iter
    (fun (i : Input.t) ->
      Fmt.pf fmt "input %a : %a@," Tensor_id.pp i.Input.id Binding.pp
        i.Input.binding)
    k.inputs;
  List.iter
    (fun (v : Value.t) ->
      Fmt.pf fmt "%a = %s(%a)@," Tensor_id.pp v.Value.id
        (Result_conversion.name v.Value.result)
        Expr.Pp.value v.Value.body)
    k.values;
  Fmt.pf fmt "outputs: %s@]"
    (comma
       (List.map
          (fun (o : Output.t) ->
            Format.asprintf "%a" Tensor_id.pp o.Output.value)
          k.outputs))
