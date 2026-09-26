type error =
  [ `Kernel of Kernel.error
  | `Lower of Loop_lower.error
  | `Unresolved_source of Tensor_id.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Kernel e -> Kernel.pp_error fmt e
  | `Lower e -> Loop_lower.pp_error fmt e
  | `Unresolved_source id ->
      Format.fprintf fmt "region program reads unresolved source %a"
        Tensor_id.pp id

(* Disjoint from every source id used in THIS mini-kernel -- [Kernel.create]'s
   own [Duplicate_id] check is scoped to its own [inputs]/[values], not the
   whole model graph, so nothing wider is needed. *)
let fresh_id ~disjoint_from =
  let max_id =
    Tensor_id.Set.fold
      (fun id acc -> max acc (Tensor_id.to_int id))
      disjoint_from (-1)
  in
  Tensor_id.of_int (max_id + 1)

(* [bindings] gives whole packed tensors, not signatures -- every source
   becomes a [Caller] input with its [Tensor_sig.t] derived from the tensor
   it is actually bound to, real operand or synthetic default alike (the two
   are indistinguishable at this point, and [Loop_lower]'s own
   [input_sources] treats [Caller]/[Captured_constant] identically, so
   nothing is lost by not preserving [region_result]'s [Filled] distinction
   here). *)
let tensor_sig_of_binding ~id (Tensor.Tensor t) =
  let quant =
    match t.Tensor.payload.Payload.quant with
    | Payload.No_quant -> None
    | Payload.Quant q -> Some q
  in
  Tensor_sig.create ~id ~name:"loop_js region source" ~shape:t.Tensor.shape
    ~fmt:(Payload.Fmt t.Tensor.payload.Payload.fmt) ?quant ()

let lower ~limits ~out_shape ~bindings program =
  let open Err.Syntax in
  let sources = Region_program.Fold.sources program in
  let source_ids =
    Expr.Source.Set.fold
      (fun s acc -> Tensor_id.Set.add (Expr_bridge.id_of_source s) acc)
      sources Tensor_id.Set.empty
  in
  let* inputs =
    Err.List.map
      (fun id ->
        match Tensor_id.Map.find_opt id bindings with
        | Some tensor ->
            Err.return
              {
                Kernel.Input.id;
                sg = tensor_sig_of_binding ~id tensor;
                binding = Kernel.Binding.Caller;
              }
        | None -> Err.fail (`Unresolved_source id))
      (Tensor_id.Set.elements source_ids)
  in
  let value_id = fresh_id ~disjoint_from:source_ids in
  let sg =
    Tensor_sig.create ~id:value_id ~name:"loop_js region target"
      ~shape:out_shape ~fmt:(Payload.Fmt Payload.F32) ()
  in
  let value =
    {
      Kernel.Value.id = value_id;
      sg;
      computation = Region_group.Ref.Solo program;
      result = Kernel.Result_conversion.Round_f32;
    }
  in
  let* kernel =
    Kernel.create ~limits ~inputs ~values:[ value ] ~outputs:[ value_id ] ()
    |> Err.map_error (fun e -> `Kernel e)
  in
  Loop_lower.lower (Fusion_plan.default kernel)
  |> Err.map_error (fun e -> `Lower e)

(* The group sibling of [lower] (T7.2): the SAME shape, but for several
   sibling values sharing one [Region_group.t] (project step 19 -- today
   only Lstm) rather than one standalone [Region_program.t]. Unlike [lower],
   no caller-supplied [~out_shape] is needed: each [selected] ordinal's own
   [Region_group.Emitter.t] already carries its [output_shape] (a group
   projects several DIFFERENTLY-shaped outputs off one shared recurrence --
   Lstm's own output/h_n/c_n are a real example -- so the shape has to live
   per-ordinal in the group itself, unlike a solo program's single result).
   [selected]'s own sources are one union, the same "every selected member's
   own [Region_group.Ref.sources]" fold [Loop_lower.lower]'s whole-graph
   [group_unit] uses -- both read the same shared locals, so both need the
   same source closure to bind them. An ordinal outside [group]'s own range
   is a caller defect ([Option.get], not a typed error), matching
   [Region_execution.materialize_group]'s own convention: every caller here
   derives [selected] from [group] itself, same as that function's callers
   do. *)
let lower_group ~limits ~bindings ~(selected : Region_group.Ordinal.t list)
    (group : Region_group.t) =
  let open Err.Syntax in
  let sources =
    List.fold_left
      (fun acc ordinal ->
        Expr.Source.Set.union acc
          (Option.get (Region_group.sources group ordinal)))
      Expr.Source.Set.empty selected
  in
  let source_ids =
    Expr.Source.Set.fold
      (fun s acc -> Tensor_id.Set.add (Expr_bridge.id_of_source s) acc)
      sources Tensor_id.Set.empty
  in
  let* inputs =
    Err.List.map
      (fun id ->
        match Tensor_id.Map.find_opt id bindings with
        | Some tensor ->
            Err.return
              {
                Kernel.Input.id;
                sg = tensor_sig_of_binding ~id tensor;
                binding = Kernel.Binding.Caller;
              }
        | None -> Err.fail (`Unresolved_source id))
      (Tensor_id.Set.elements source_ids)
  in
  let first_id = fresh_id ~disjoint_from:source_ids in
  let values =
    List.mapi
      (fun i ordinal ->
        let emitter = Option.get (Region_group.emitter group ordinal) in
        let id = Tensor_id.of_int ((first_id :> int) + i) in
        let sg =
          Tensor_sig.create ~id ~name:"loop_js region target"
            ~shape:emitter.Region_group.Emitter.output_shape
            ~fmt:(Payload.Fmt Payload.F32) ()
        in
        {
          Kernel.Value.id;
          sg;
          computation = Region_group.Ref.Grouped (group, ordinal);
          result = Kernel.Result_conversion.Round_f32;
        })
      selected
  in
  let* kernel =
    Kernel.create ~limits ~inputs ~values
      ~outputs:(List.map (fun (v : Kernel.Value.t) -> v.id) values)
      ()
    |> Err.map_error (fun e -> `Kernel e)
  in
  let+ program =
    Loop_lower.lower (Fusion_plan.default kernel)
    |> Err.map_error (fun e -> `Lower e)
  in
  ( program,
    List.map2
      (fun ordinal (v : Kernel.Value.t) -> (ordinal, v.Kernel.Value.id))
      selected values )
