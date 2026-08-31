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
    computation : Region_program.t;
    result : Result_conversion.t;
  }
end

module Output = struct
  type t = { value : Tensor_id.t; sg : Tensor_sig.t }
end

module Use = struct
  type t = { producer : Tensor_id.t; consumer : Tensor_id.t }

  let compare a b =
    match Tensor_id.compare a.producer b.producer with
    | 0 -> Tensor_id.compare a.consumer b.consumer
    | c -> c

  let pp fmt u =
    Fmt.pf fmt "%a->%a" Tensor_id.pp u.producer Tensor_id.pp u.consumer

  module Set = Set.Make (struct
    type nonrec t = t

    let compare = compare
  end)
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

    (* Measured under node with [Kernel_eval.value_at] over a real producer
       chain (test/native/depth_probe.ml). The frontier there is both lower and
       less stable than for a flat expression -- a 1024-transition chain
       overflowed on three runs out of four at the previous ceiling, and a
       384-transition chain of depth-4 bodies overflows while a 192-transition
       chain of depth-16 bodies does not, so transition count dominates and the
       limit does not fit a tidy cost model. Region execution classification
       adds a small fixed frame cost, so 96 preserves headroom for the mixed
       producer/body frontier on both native and JavaScript backends. *)
    let eval_recursion = 96

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
      Err.fail (`Invalid_limit { Invalid.name; value = Int64.of_int v })
    else Err.return ()

  let check_int64 name v hard =
    if Int64.compare v 0L <= 0 || Int64.compare v hard >= 0 then
      Err.fail (`Invalid_limit { Invalid.name; value = v })
    else Err.return ()

  let create ~max_size ~max_depth ~max_values ~max_dep_depth ~max_inputs
      ~max_outputs ~max_extent ~max_numel =
    let open Err.Syntax in
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
    Err.or_raise ~pp_error
      (create ~max_size:4096 ~max_depth:128 ~max_values:4096 ~max_dep_depth:1024
         ~max_inputs:1024 ~max_outputs:1024 ~max_extent:0x7FFF_FFFFL
         ~max_numel:0x7FFF_FFFFL)
end

type t = {
  inputs : Input.t list;
  values : Value.t list;
  outputs : Output.t list;
  limits : Limits.t;
  by_id : Value.t Tensor_id.Map.t;
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
  type t = { at : Tensor_id.t; error : Region_program.error }
end

type error =
  [ `Body of Body_error.t
  | `Dependency_too_deep of int
  | `Duplicate_id of Tensor_id.t
  | `Eval_too_deep of int
  | `Extent_too_large of Extent_bound.t
  | `Forward_reference of Forward_ref.t
  | `Not_materializable of Format_rule.t
  | `Numel_too_large of Tensor_id.t
  | `Quant_contract of Tensor_id.t
  | `Signature_id_mismatch of Sig_mismatch.t
  | `Too_many_inputs of int
  | `Too_many_outputs of int
  | `Too_many_values of int
  | `Unknown_output of Tensor_id.t
  | `Unreachable_value of Tensor_id.t
  | `Unresolved_source of Unresolved.t ]

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
      Fmt.pf fmt "%a: %a" Tensor_id.pp at Region_program.pp_error error

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
    let open Err.Syntax in
    let* () =
      List.fold_left
        (fun acc axis ->
          let* () = acc in
          let e =
            Int64.of_int (Dim.to_int (Vec6.get sg.Tensor_sig.shape axis))
          in
          if Int64.compare e limits.Limits.max_extent > 0 then
            Err.fail (`Extent_too_large { Extent_bound.id; axis; extent = e })
          else Err.return ())
        (Err.return ()) Expr.Axis.all
    in
    (* [max_numel] is a configured limit and INCLUSIVE ([Limits.create] checks
       it stays BELOW [Hard.numel], never at or above); [Vec6.numel_bounded]'s
       [~limit] is EXCLUSIVE -- the smallest rejected count, matching every
       other [Hard.*] comparison in the engine. The [succ] converts the
       convention once, here, rather than tightening the configured limit by
       one element; it cannot overflow because [max_numel < Hard.numel]. *)
    let+ _ =
      Err.map_error
        (fun (`Numel_over_limit _) -> `Numel_too_large id)
        (Vec6.numel_bounded
           ~limit:(Int64.succ limits.Limits.max_numel)
           sg.Tensor_sig.shape)
    in
    ()
end

(* ---- signature contracts -------------------------------------------------- *)

let quant_contract id (sg : Tensor_sig.t) =
  let quantized = Payload.is_quantized sg.Tensor_sig.fmt in
  match (quantized, sg.Tensor_sig.quant) with
  | false, None -> Err.return ()
  | true, Some q -> (
      (* A per-channel value's two arrays agree by [Quant.per_channel]'s
         construction; relating that common length to C is this side's job. *)
      match Quant.channel_count q with
      | None -> Err.return ()
      | Some n ->
          if n = Dim.to_int (Vec6.get sg.Tensor_sig.shape Expr.Axis.C) then
            Err.return ()
          else Err.fail (`Quant_contract id))
  | true, None | false, Some _ -> Err.fail (`Quant_contract id)

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
  if f32 && Option.is_none sg.Tensor_sig.quant then Err.return ()
  else
    Err.fail
      (`Not_materializable { Format_rule.id; role; fmt = sg.Tensor_sig.fmt })

(* ---- construction --------------------------------------------------------- *)

(* Does [l] hold more than [limit] cells? Stops one past the limit instead of
   walking to the end: [List.length] on an untrusted list bounds the work that
   FOLLOWS the check but not the work the check itself does, which is the same
   budgets-before-work rule applied to the guard. Never forms a sum, either — a
   count added to another count is an unchecked [int] aggregate, and under
   js_of_ocaml a wrapped negative would sail past a [> limit] test. *)
let over_limit limit l =
  (* A negative limit is exceeded by ANY list, the empty one included: zero
     cells is more than a negative count. Every production caller passes a
     validated positive [Limits] field, but the helper is public and its
     contract has to hold on its own terms rather than on its callers'. *)
  if limit < 0 then true
  else
    let rec go n = function
      | [] -> false
      | _ :: tl -> if n >= limit then true else go (n + 1) tl
    in
    go 0 l

(* Continue counting a second list against the capacity the first left, so the
   two are bounded together without adding them. *)
let over_limit_2 limit a b =
  if limit < 0 then true
  else
    let rec go n = function
      | [] -> (n, false)
      | _ :: tl -> if n >= limit then (n, true) else go (n + 1) tl
    in
    match go 0 a with
    | _, true -> true
    | n, false -> ( match go n b with _, over -> over)

let create ?(limits = Limits.default) ~inputs ~values ~outputs () =
  let open Err.Syntax in
  (* Arity first: the cheapest guards, and they bound the list and map work
     every later check performs. Neither list contributes to [max_values], so a
     kernel with a thousand unused inputs would otherwise satisfy every DAG
     budget while imposing unbounded setup on the evaluator, which validates
     each declared input before computing anything. *)
  let* () =
    if over_limit limits.Limits.max_inputs inputs then
      Err.fail (`Too_many_inputs limits.Limits.max_inputs)
    else Err.return ()
  in
  let* () =
    if over_limit limits.Limits.max_outputs outputs then
      Err.fail (`Too_many_outputs limits.Limits.max_outputs)
    else Err.return ()
  in
  let* () =
    if over_limit limits.Limits.max_values values then
      Err.fail (`Too_many_values limits.Limits.max_values)
    else Err.return ()
  in
  (* Budgets before any unmetered traversal. [Expr.Fold]'s queries walk the whole
     tree; running one first would exhaust the stack on precisely the oversized
     body the limit exists to reject. *)
  let* () =
    List.fold_left
      (fun acc (v : Value.t) ->
        let* () = acc in
        (* [Err.map_error], not a hand-rolled rebuild: it preserves the
           original detection backtrace, which unwrapping [.kind] and calling
           [Err.fail] silently would not. *)
        Err.map_error
          (fun e -> `Body { Body_error.at = v.id; error = e })
          (Region_program.check ~max_size:limits.Limits.max_size
             ~max_depth:limits.Limits.max_depth v.computation))
      (Err.return ()) values
  in
  (* Identity: the record id and its signature id must agree, since [sg.id] is
     the binding key everywhere else. Leaving them independent would key source
     resolution and binding differently. *)
  let* () =
    let check id (sg : Tensor_sig.t) =
      if Tensor_id.equal id sg.Tensor_sig.id then Err.return ()
      else
        Err.fail
          (`Signature_id_mismatch
             { Sig_mismatch.record = id; sg = sg.Tensor_sig.id })
    in
    let* () =
      List.fold_left
        (fun acc (i : Input.t) ->
          let* () = acc in
          check i.id i.sg)
        (Err.return ()) inputs
    in
    List.fold_left
      (fun acc (v : Value.t) ->
        let* () = acc in
        check v.id v.sg)
      (Err.return ()) values
  in
  let* _seen =
    List.fold_left
      (fun acc id ->
        let* seen = acc in
        if Tensor_id.Set.mem id seen then Err.fail (`Duplicate_id id)
        else Err.return (Tensor_id.Set.add id seen))
      (Err.return Tensor_id.Set.empty)
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
              if Tensor_id.Set.mem id input_ids then Err.return (d, e)
              else
                match
                  (Tensor_id.Map.find_opt id dep, Tensor_id.Map.find_opt id ev)
                with
                | Some pd, Some pe -> Err.return (max d pd, max e pe)
                | _ ->
                    (* Defined later, or not at all: the ordered list makes
                       these the same walk. A source naming a value that exists
                       further down is a forward reference; anything else is
                       unresolved. *)
                    if Tensor_id.Set.mem id defined then
                      Err.fail
                        (`Forward_reference
                           { Forward_ref.at = v.Value.id; depends_on = id })
                    else
                      Err.fail
                        (`Unresolved_source
                           { Unresolved.at = v.Value.id; source = src }))
            (Region_program.Fold.sources v.Value.computation)
            (Err.return (0, 0))
        in
        (* The CONVERTED body, not the raw one: every consumer — a store, a
           load, [value_at] — evaluates [Result_conversion.apply], so the
           conversion node is a level the evaluator really walks and measuring
           [v.body] undercounts each value by it.

           This bound covers expression levels only. The per-producer-transition
           cost, which dominates and which measurement showed does not fit a
           weighted sum, is bounded at runtime by [Hard.eval_recursion] instead
           of being folded into a static weight here. *)
        let d = d + 1
        and e = e + 1 + Region_program.Fold.max_depth v.Value.computation in
        let* () =
          if d > limits.Limits.max_dep_depth then
            Err.fail (`Dependency_too_deep limits.Limits.max_dep_depth)
          else Err.return ()
        in
        let* () =
          if e > Limits.Hard.eval_depth then
            Err.fail (`Eval_too_deep Limits.Hard.eval_depth)
          else Err.return ()
        in
        Err.return
          ( Tensor_id.Map.add v.Value.id d dep,
            Tensor_id.Map.add v.Value.id e ev,
            defined ))
      (Err.return
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
        | None -> Err.fail (`Unknown_output id)
        | Some sg -> Err.return ({ Output.value = id; sg } :: out))
      (Err.return []) outputs
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
              (Region_program.Fold.sources v.Value.computation)
              live
          else live)
        live (List.rev values)
    in
    List.fold_left
      (fun acc (v : Value.t) ->
        let* () = acc in
        if Tensor_id.Set.mem v.Value.id live then Err.return ()
        else Err.fail (`Unreachable_value v.Value.id))
      (Err.return ()) values
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
          | Binding.Caller | Binding.Captured_constant -> Err.return ()
        in
        Bounds.signature limits i.Input.id i.Input.sg)
      (Err.return ()) inputs
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
      (Err.return ()) values
  in
  Err.return
    {
      inputs;
      values;
      outputs = out;
      limits;
      by_id =
        List.fold_left
          (fun m (v : Value.t) -> Tensor_id.Map.add v.Value.id v m)
          Tensor_id.Map.empty values;
    }

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
      let body =
        match Region_program.pixel_expression v.Value.computation with
        | Some body -> Core.Pretty.to_string Expr.Pp.value body
        | None -> Core.Pretty.to_string Region_program.pp v.Value.computation
      in
      Fmt.pf fmt "%a = %s(%s)@," Tensor_id.pp v.Value.id
        (Result_conversion.name v.Value.result)
        body)
    k.values;
  Fmt.pf fmt "outputs: %s@]"
    (comma
       (List.map
          (fun (o : Output.t) ->
            Format.asprintf "%a" Tensor_id.pp o.Output.value)
          k.outputs))

(* Indexed, so O(log values) through the balanced map [create] built — not O(1),
   which a persistent [Map.Make] cannot give. What matters is that the whole-list
   scan is gone: it was multiplied by every caller resolving an edge's endpoints,
   and the planner does so per candidate, making it an O(candidates x values)
   term. A hash table would be constant expected time but would trade away the
   immutability the rest of the representation relies on. *)
let value (k : t) id = Tensor_id.Map.find_opt id k.by_id

(* Both edge sets restrict to sources resolving to a VALUE: a load of a boundary
   input is a dependency of the body but not an edge anything may virtualize.

   Membership through a set built once, not [value]'s linear scan per source:
   that made edge derivation quadratic in the value count before any planning
   had begun. *)
let edges_of (k : t) project =
  let ids =
    List.fold_left
      (fun s (v : Value.t) -> Tensor_id.Set.add v.Value.id s)
      Tensor_id.Set.empty k.values
  in
  let is_value id = Tensor_id.Set.mem id ids in
  List.fold_left
    (fun acc (v : Value.t) ->
      List.fold_left
        (fun acc src ->
          let producer = Expr_bridge.id_of_source src in
          if is_value producer then
            Use.Set.add { Use.producer; consumer = v.Value.id } acc
          else acc)
        acc
        (project v.Value.computation))
    Use.Set.empty k.values

let uses k =
  edges_of k (fun computation ->
      Expr.Source.Set.elements (Region_program.Fold.sources computation))

let load_uses k =
  edges_of k (fun computation ->
      List.map fst (Region_program.Fold.loads computation))

let pixel_expression (v : Value.t) =
  Region_program.pixel_expression v.computation
