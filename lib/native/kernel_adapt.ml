(* See kernel_adapt.mli. *)

module Unknown_stage = struct
  type t = { at : Tensor_id.t; source : Expr.Source.t }
end

module Program_error = struct
  type kind = Duplicate_definition | Signature_id | Forward_source
  type t = { at : Tensor_id.t; kind : kind }

  let kind_name = function
    | Duplicate_definition -> "defined twice"
    | Signature_id -> "signature id disagrees"
    | Forward_source -> "reads a later stage"
end

type error =
  [ Kernel.error
  | `Unknown_stage_source of Unknown_stage.t
  | `Missing_live_output of Tensor_id.t
  | `Output_not_selected of Tensor_id.t
  | `Unknown_selection of Tensor_id.t
  | `Passthrough_output of Tensor_id.t
  | `Unknown_program_output of Tensor_id.t
  | `Program_invalid of Program_error.t ]

let pp_error fmt : [< error ] -> unit = function
  | `Unknown_stage_source { Unknown_stage.at; source } ->
      Fmt.pf fmt "stage %a reads unknown source %a" Tensor_id.pp at Tensor_id.pp
        (Expr_bridge.id_of_source source)
  | `Missing_live_output id ->
      (* Three categories reach here — a graph output, a value used outside the
         selection, and a dead terminal — so the message names the requirement
         rather than guessing which one applies. *)
      Fmt.pf fmt "outputs must begin with the required list; expected %a next"
        Tensor_id.pp id
  | `Output_not_selected id ->
      Fmt.pf fmt "output %a is not a selected stage" Tensor_id.pp id
  | `Unknown_selection id ->
      Fmt.pf fmt "selection names %a, which is not a stage" Tensor_id.pp id
  | `Passthrough_output id ->
      Fmt.pf fmt "graph output %a is a boundary input, not a stage result"
        Tensor_id.pp id
  | `Unknown_program_output id ->
      Fmt.pf fmt "graph output %a is neither a stage nor a boundary"
        Tensor_id.pp id
  | `Program_invalid { Program_error.at; kind } ->
      Fmt.pf fmt "stage program invalid at %a: %s" Tensor_id.pp at
        (Program_error.kind_name kind)
  | #Kernel.error as e -> Kernel.pp_error fmt e

(* ---- the shared checked analysis ------------------------------------------

   Both public entry points go through here, and it runs the guards before the
   traversals they guard. *)

type analysis = {
  select : Tensor_id.Set.t;
  kinds : Graph_common.Input.kind Tensor_id.Map.t;
  sources : Expr.Source.Set.t Tensor_id.Map.t;  (** per stage, memoised *)
  stage_sig : Tensor_sig.t Tensor_id.Map.t;
  order : Tensor_id.t list;  (** stage ids, topological *)
}

let analyse ~limits ~select (p : Stage_program.t) =
  let open Err.Syntax in
  (* Every untrusted raw list is bounded BEFORE it is traversed — not only the
     stages. [inputs], [consts] and [outputs] are just as public, and leaving
     them unbounded meant the adapter did unbounded work and only [Kernel.create]
     noticed, on the DERIVED interface, after all analysis had run. Bounding the
     raw lists here also makes the two entry points agree: [required_outputs]
     never reaches [Kernel.create] at all.

     Raw inputs and consts are both bounded by [max_inputs] because both become
     boundary inputs; the derived interface is bounded again by [create], which
     is the tighter statement and the one that counts synthetic entries. *)
  (* [Kernel.over_limit], not [List.length]: an over-limit list must stop one
     cell past the limit rather than be walked to its end, or the guard is
     itself the unbounded work it exists to prevent. Inputs and consts are
     counted TOGETHER by continuing the second from the first's remainder — the
     sum of two counts is an unchecked [int] aggregate, and under js_of_ocaml a
     wrapped negative would sail straight past a [> limit] test. *)
  let* () =
    if Kernel.over_limit limits.Kernel.Limits.max_values p.Stage_program.stages
    then Err.fail (`Too_many_values limits.Kernel.Limits.max_values)
    else Err.return ()
  in
  let* () =
    if
      Kernel.over_limit_2 limits.Kernel.Limits.max_inputs p.Stage_program.inputs
        p.Stage_program.consts
    then Err.fail (`Too_many_inputs limits.Kernel.Limits.max_inputs)
    else Err.return ()
  in
  let* () =
    if
      Kernel.over_limit limits.Kernel.Limits.max_outputs p.Stage_program.outputs
    then Err.fail (`Too_many_outputs limits.Kernel.Limits.max_outputs)
    else Err.return ()
  in
  (* Then every body's budget — selected or not. An oversized UNSELECTED body is
     just as dangerous: liveness scans it. *)
  let* () =
    List.fold_left
      (fun acc (st : Stage_program.Stage.t) ->
        let* () = acc in
        Err.map_error
          (fun e -> `Body { Kernel.Body_error.at = st.id; error = e })
          (Expr.Check.value ~max_size:limits.Kernel.Limits.max_size
             ~max_depth:limits.Kernel.Limits.max_depth st.body))
      (Err.return ()) p.Stage_program.stages
  in
  (* The boundary table is built RESULT-VALUED, not with [Map.add]. Insertion
     silently overwrote a duplicate input, a duplicate constant, or an
     input/constant id collision, and checked no signature id — so a selection
     could still launder a malformed boundary definition, which is precisely
     what treating [Stage_program.t] as untrusted is supposed to stop. *)
  let* boundary =
    let add m id (sg : Tensor_sig.t) =
      let invalid kind =
        Err.fail (`Program_invalid { Program_error.at = id; kind })
      in
      if Tensor_id.Map.mem id m then invalid Program_error.Duplicate_definition
      else if not (Tensor_id.equal id sg.Tensor_sig.id) then
        invalid Program_error.Signature_id
      else Err.return (Tensor_id.Map.add id sg m)
    in
    let* m =
      List.fold_left
        (fun acc (id, sg) ->
          let* m = acc in
          add m id sg)
        (Err.return Tensor_id.Map.empty)
        p.Stage_program.inputs
    in
    List.fold_left
      (fun acc ((sg : Tensor_sig.t), _) ->
        let* m = acc in
        add m sg.Tensor_sig.id sg)
      (Err.return m) p.Stage_program.consts
  in
  let stage_sig =
    List.fold_left
      (fun m (st : Stage_program.Stage.t) -> Tensor_id.Map.add st.id st.sg m)
      Tensor_id.Map.empty p.Stage_program.stages
  in
  (* Validate the whole definition table before projecting any selection.
     Otherwise a selection launders a structural defect: with stage [a] reading
     a later stage [b], selecting only [a] turns [b] into a synthetic boundary
     input, and [Kernel.create]'s forward-reference rule never sees it. *)
  let* sources, _ =
    List.fold_left
      (fun acc (st : Stage_program.Stage.t) ->
        let* sources, defined = acc in
        let invalid kind =
          Err.fail (`Program_invalid { Program_error.at = st.id; kind })
        in
        let* () =
          if Tensor_id.Map.mem st.id sources || Tensor_id.Map.mem st.id boundary
          then invalid Program_error.Duplicate_definition
          else Err.return ()
        in
        let* () =
          if Tensor_id.equal st.id st.sg.Tensor_sig.id then Err.return ()
          else invalid Program_error.Signature_id
        in
        let srcs = Expr.Fold.sources st.body in
        let* () =
          Expr.Source.Set.fold
            (fun src acc ->
              let* () = acc in
              let id = Expr_bridge.id_of_source src in
              if Tensor_id.Map.mem id boundary || Tensor_id.Set.mem id defined
              then Err.return ()
              else if Tensor_id.Map.mem id stage_sig then
                invalid Program_error.Forward_source
              else
                Err.fail
                  (`Unknown_stage_source
                     { Unknown_stage.at = st.id; source = src }))
            srcs (Err.return ())
        in
        Err.return
          (Tensor_id.Map.add st.id srcs sources, Tensor_id.Set.add st.id defined))
      (Err.return (Tensor_id.Map.empty, Tensor_id.Set.empty))
      p.Stage_program.stages
  in
  (* Every declared output is classified HERE, before any selection: a stage id
     is fine, a boundary id is a pass-through, and an id in neither table is
     malformed. Leaving it to the selection-dependent pass let an unknown id be
     silently skipped, so a malformed program adapted cleanly to a kernel with a
     DIFFERENT public result — the very hazard the pass-through rule exists to
     prevent. Neither classification depends on the selection, so neither
     belongs after it. *)
  let* () =
    List.fold_left
      (fun acc id ->
        let* () = acc in
        if Tensor_id.Map.mem id stage_sig then Err.return ()
        else if Tensor_id.Map.mem id boundary then
          Err.fail (`Passthrough_output id)
        else Err.fail (`Unknown_program_output id))
      (Err.return ()) p.Stage_program.outputs
  in
  let order =
    List.map
      (fun (st : Stage_program.Stage.t) -> st.Stage_program.Stage.id)
      p.Stage_program.stages
  in
  (* An id in [select] naming no stage is an error, so the optional set has
     exact rather than best-effort semantics. *)
  let all_stages =
    List.fold_left
      (fun s id -> Tensor_id.Set.add id s)
      Tensor_id.Set.empty order
  in
  let select = Option.value select ~default:all_stages in
  let* () =
    Tensor_id.Set.fold
      (fun id acc ->
        let* () = acc in
        if Tensor_id.Set.mem id all_stages then Err.return ()
        else Err.fail (`Unknown_selection id))
      select (Err.return ())
  in
  Err.return
    { select; kinds = p.Stage_program.input_kinds; sources; stage_sig; order }

let sources_of a id =
  Option.value
    (Tensor_id.Map.find_opt id a.sources)
    ~default:Expr.Source.Set.empty

(* Selected values that something outside the selection needs. *)
let externally_used a =
  List.fold_left
    (fun live id ->
      if Tensor_id.Set.mem id a.select then live
      else
        Expr.Source.Set.fold
          (fun src live ->
            let s = Expr_bridge.id_of_source src in
            if Tensor_id.Set.mem s a.select then Tensor_id.Set.add s live
            else live)
          (sources_of a id) live)
    Tensor_id.Set.empty a.order

(* Selected values nothing consumes, inside the selection or out. *)
let dead_terminals a =
  let consumed =
    List.fold_left
      (fun s id ->
        Expr.Source.Set.fold
          (fun src s -> Tensor_id.Set.add (Expr_bridge.id_of_source src) s)
          (sources_of a id) s)
      Tensor_id.Set.empty a.order
  in
  List.filter
    (fun id ->
      Tensor_id.Set.mem id a.select && not (Tensor_id.Set.mem id consumed))
    a.order

let required a (p : Stage_program.t) =
  let open Err.Syntax in
  (* Graph outputs first, order and repeats preserved. A graph output that is a
     boundary input has no value to name, so it is rejected rather than
     dropped. *)
  let* graph_outs =
    List.fold_left
      (fun acc id ->
        let* outs = acc in
        (* Classified already; an unselected stage is simply not this kernel's
           output. *)
        if Tensor_id.Set.mem id a.select then Err.return (id :: outs)
        else Err.return outs)
      (Err.return []) p.Stage_program.outputs
  in
  let graph_outs = List.rev graph_outs in
  let seen =
    List.fold_left
      (fun s id -> Tensor_id.Set.add id s)
      Tensor_id.Set.empty graph_outs
  in
  let append (acc, seen) id =
    if Tensor_id.Set.mem id seen then (acc, seen)
    else (id :: acc, Tensor_id.Set.add id seen)
  in
  let live = externally_used a in
  let acc, seen =
    List.fold_left
      (fun st id -> if Tensor_id.Set.mem id live then append st id else st)
      ([], seen) a.order
  in
  let acc, _ = List.fold_left append (acc, seen) (dead_terminals a) in
  Err.return (graph_outs @ List.rev acc)

let required_outputs ?(limits = Kernel.Limits.default) ?select p =
  let open Err.Syntax in
  let* a = analyse ~limits ~select p in
  let* outs = required a p in
  (* The DERIVED bound, enforced here too: this entry point never reaches
     [Kernel.create], so without it the same limits would mean different things
     depending on which function a caller used. *)
  if Kernel.over_limit limits.Kernel.Limits.max_outputs outs then
    Err.fail (`Too_many_outputs limits.Kernel.Limits.max_outputs)
  else Err.return outs

let of_stage_program ?(limits = Kernel.Limits.default) ?select ?outputs p =
  let open Err.Syntax in
  let* a = analyse ~limits ~select p in
  let* req = required a p in
  (* Caller outputs must BEGIN with the required sequence, order and repeats
     included; extras follow and must name selected values. *)
  let* outputs =
    match outputs with
    | None -> Err.return req
    | Some given ->
        (* The caller's list is a public collection too, and was prefix-scanned
           and then extra-scanned before [Kernel.create] ever counted it. *)
        let* () =
          if Kernel.over_limit limits.Kernel.Limits.max_outputs given then
            Err.fail (`Too_many_outputs limits.Kernel.Limits.max_outputs)
          else Err.return ()
        in
        let rec strip req given =
          match (req, given) with
          | [], rest -> Err.return rest
          | r :: rt, g :: gt when Tensor_id.equal r g -> strip rt gt
          | r :: _, _ -> Err.fail (`Missing_live_output r)
        in
        let* extras = strip req given in
        let+ () =
          List.fold_left
            (fun acc id ->
              let* () = acc in
              if Tensor_id.Set.mem id a.select then Err.return ()
              else Err.fail (`Output_not_selected id))
            (Err.return ()) extras
        in
        given
  in
  (* Boundary inputs are DERIVED from the selected bodies' free sources. Copying
     every program input and const would make the evaluator demand bindings for
     an unrelated branch — it validates each declared input before computing
     anything — and materialise unrelated fills for nothing. *)
  let referenced =
    List.fold_left
      (fun s id ->
        if Tensor_id.Set.mem id a.select then
          Expr.Source.Set.fold
            (fun src s -> Tensor_id.Set.add (Expr_bridge.id_of_source src) s)
            (sources_of a id) s
        else s)
      Tensor_id.Set.empty a.order
  in
  let kind_of id =
    (* The sparse-map default, the rule [Graph_common.input_kind] exists to
       state: a total lookup would silently reclassify an unlisted input. *)
    Option.value
      (Tensor_id.Map.find_opt id a.kinds)
      ~default:Graph_common.Input.Input
  in
  let inputs =
    (* Original inputs and consts in [Stage_program] order, retained only where
       referenced; then one deduplicated synthetic input per referenced
       unselected stage. *)
    let keep =
      List.filter_map
        (fun (id, sg) ->
          if Tensor_id.Set.mem id referenced then
            Some
              {
                Kernel.Input.id;
                sg;
                binding =
                  (match kind_of id with
                  | Graph_common.Input.Input -> Kernel.Binding.Caller
                  | Graph_common.Input.Constant ->
                      Kernel.Binding.Captured_constant);
              }
          else None)
        p.Stage_program.inputs
    in
    let consts =
      List.filter_map
        (fun ((sg : Tensor_sig.t), v) ->
          if Tensor_id.Set.mem sg.id referenced then
            Some
              { Kernel.Input.id = sg.id; sg; binding = Kernel.Binding.Filled v }
          else None)
        p.Stage_program.consts
    in
    let outside =
      List.filter_map
        (fun id ->
          if
            Tensor_id.Set.mem id referenced
            && not (Tensor_id.Set.mem id a.select)
          then
            Option.map
              (fun sg ->
                { Kernel.Input.id; sg; binding = Kernel.Binding.Caller })
              (Tensor_id.Map.find_opt id a.stage_sig)
          else None)
        a.order
    in
    keep @ consts @ outside
  in
  let values =
    List.filter_map
      (fun (st : Stage_program.Stage.t) ->
        if Tensor_id.Set.mem st.id a.select then
          Some
            {
              Kernel.Value.id = st.id;
              sg = st.sg;
              body = st.body;
              result = Kernel.Result_conversion.Round_f32;
            }
        else None)
      p.Stage_program.stages
  in
  Err.map_error
    (fun (e : Kernel.error) -> (e :> error))
    (Kernel.create ~limits ~inputs ~values ~outputs ())
