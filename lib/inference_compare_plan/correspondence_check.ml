open Inference_compare_schema
module Codecs = Inference_compare_codecs
module Recipe_eval = Codecs.Recipe_eval

type options = Recipe_scope.options = {
  codecs : Codecs.Element_codec.Registry.t;
  operations : Codecs.Conversion_registry.t;
}

module Unit_plan = struct
  type t = {
    unit : Id.Unit.t;
    canonical_reads : Id.Canonical_value.t list;
    canonical_writes : Id.Canonical_value.t list;
    input_bindings : Input_binding.t list;
    input_checks : Input_check.t list;
    output_bindings : Output_binding.t list;
    output_pairs : Member_pair.t list;
  }
end

module Key_map = Map.Make (struct
  type t = Id.Canonical_value.t

  let compare = Id.Canonical_value.compare
end)

module Unit_map = Map.Make (struct
  type t = Id.Unit.t

  let compare = Id.Unit.compare
end)

module Unit_side_map = Map.Make (struct
  type t = Id.Unit.t * Side.t

  let compare (u, s) (u', s') =
    match Id.Unit.compare u u' with 0 -> Side.compare s s' | c -> c
end)

module Str_set = Set.Make (String)

type t = {
  plans : Unit_plan.t list;
  writers : Id.Unit.t Key_map.t;
  initials : Id.Canonical_value.t list;
  specs : Tensor_spec.t Key_map.t;
  finals : Id.Canonical_value.t list;
  coverage : Coverage.t;
}

open Recipe_scope

let index_of ~left ~right = function Side.Left -> left | Side.Right -> right

let check ~options ~left ~right ~partition (m : Mapping_artifact.t) =
  let frontier = m.frontier in
  let idx side = index_of ~left ~right side in
  let tensor_spec side t = Graph_index.spec (idx side) t in
  (* Canonical values: specs, initial bindings. *)
  let specs =
    List.fold_left
      (fun acc (c : Canonical_spec.t) -> Key_map.add c.key c.spec acc)
      Key_map.empty frontier.canonical_values
  in
  let canonical_env keys : Id.Canonical_value.t -> Tensor_spec.t option =
   fun k ->
    if List.exists (Id.Canonical_value.equal k) keys then
      Key_map.find_opt k specs
    else None
  in
  let all_keys = List.map fst (Key_map.bindings specs) in
  let initials =
    List.map (fun (b : Initial_binding.t) -> b.key) frontier.initial_values
  in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | k :: rest ->
          if Key_map.mem k specs then go rest
          else fail (Plan_error.make ~canonical:k Unknown_member)
    in
    go initials
  in
  let unit_exists u = Partition_check.matched_unit partition u <> None in
  (* Input bindings. *)
  let bindings_for unit side =
    List.filter
      (fun (b : Input_binding.t) ->
        Id.Unit.equal b.unit unit && Side.equal b.side side)
      frontier.input_bindings
  in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (b : Input_binding.t) :: rest ->
          if not (unit_exists b.unit) then
            fail
              (Plan_error.make ~side:b.side ~unit:b.unit ~tensor:b.tensor
                 Unknown_member)
          else go rest
    in
    go frontier.input_bindings
  in
  let binding_reads = ref [] in
  let* reads_by_unit =
    let rec per_boundary acc = function
      | [] -> Ok acc
      | (bd : Partition_check.Boundary.t) :: rest ->
          let bs = bindings_for bd.unit bd.side in
          (* Every binding names an input of this unit; every input has
             exactly one binding. *)
          let rec bound_inputs = function
            | [] -> Ok ()
            | (b : Input_binding.t) :: more ->
                if List.exists (Tensor_ref.equal b.tensor) bd.inputs then
                  bound_inputs more
                else
                  fail
                    (Plan_error.make ~side:bd.side ~unit:bd.unit
                       ~tensor:b.tensor Unknown_member)
          in
          let* () = bound_inputs bs in
          let rec each_input acc = function
            | [] -> Ok acc
            | (t : Tensor_ref.t) :: more -> (
                match
                  List.filter
                    (fun (b : Input_binding.t) -> Tensor_ref.equal b.tensor t)
                    bs
                with
                | [] ->
                    fail
                      (Plan_error.make ~side:bd.side ~unit:bd.unit ~tensor:t
                         Missing_common_input)
                | _ :: _ :: _ ->
                    fail
                      (Plan_error.make ~side:bd.side ~unit:bd.unit ~tensor:t
                         Ambiguous_route)
                | [ b ] ->
                    let* expected =
                      match tensor_spec bd.side t with
                      | Some s -> Ok s
                      | None ->
                          fail
                            (Plan_error.make ~side:bd.side ~unit:bd.unit
                               ~tensor:t Unknown_member)
                    in
                    let* checked =
                      check_recipe ~options ~canonical:(canonical_env all_keys)
                        ~boundary:(fun _ -> None)
                        ~side:bd.side ~unit:bd.unit ~tensor:t b.from_common
                    in
                    let* () =
                      require_lossless ~side:bd.side ~unit:bd.unit ~tensor:t
                        checked
                    in
                    let* () =
                      require_spec ~side:bd.side ~unit:bd.unit ~tensor:t
                        ~expected checked
                    in
                    binding_reads :=
                      ( bd.unit,
                        bd.side,
                        t,
                        Recipe_eval.Checked.canonical_reads checked )
                      :: !binding_reads;
                    each_input
                      (Recipe_eval.Checked.canonical_reads checked @ acc)
                      more)
          in
          let* reads = each_input [] bd.inputs in
          per_boundary
            (Unit_side_map.update (bd.unit, bd.side)
               (fun l -> Some (reads @ Option.value l ~default:[]))
               acc)
            rest
    in
    per_boundary Unit_side_map.empty (Partition_check.boundaries partition)
  in
  (* A side must reconstruct exactly the canonical values its own bindings
     read: a value only the other side consumes needs no imports here, and
     could not be reconstructed from none. *)
  let reads_of_side u side =
    List.sort_uniq Id.Canonical_value.compare
      (Option.value
         (Unit_side_map.find_opt (u, side) reads_by_unit)
         ~default:[])
  in
  let reads_of u =
    List.sort_uniq Id.Canonical_value.compare
      (reads_of_side u Side.Left @ reads_of_side u Side.Right)
  in
  (* Input checks: one per (unit, side, canonical read). *)
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (c : Input_check.t) :: rest ->
          if not (unit_exists c.unit) then
            fail
              (Plan_error.make ~side:c.side ~unit:c.unit ~canonical:c.canonical
                 Unknown_member)
          else if
            not
              (List.exists
                 (Id.Canonical_value.equal c.canonical)
                 (reads_of_side c.unit c.side))
          then
            fail
              (Plan_error.make ~side:c.side ~unit:c.unit ~canonical:c.canonical
                 Unknown_member)
          else go rest
    in
    go frontier.input_checks
  in
  let* () =
    let rec per_boundary = function
      | [] -> Ok ()
      | (bd : Partition_check.Boundary.t) :: rest ->
          let boundary_env =
            tensor_env
              (List.filter_map
                 (fun t -> Option.map (fun s -> (t, s)) (tensor_spec bd.side t))
                 bd.inputs)
          in
          let rec per_read = function
            | [] -> Ok ()
            | k :: more -> (
                match
                  List.filter
                    (fun (c : Input_check.t) ->
                      Id.Unit.equal c.unit bd.unit
                      && Side.equal c.side bd.side
                      && Id.Canonical_value.equal c.canonical k)
                    frontier.input_checks
                with
                | [] ->
                    fail
                      (Plan_error.make ~side:bd.side ~unit:bd.unit ~canonical:k
                         Missing_common_input)
                | _ :: _ :: _ ->
                    fail
                      (Plan_error.make ~side:bd.side ~unit:bd.unit ~canonical:k
                         Ambiguous_route)
                | [ c ] ->
                    let* checked =
                      check_recipe ~options
                        ~canonical:(fun _ -> None)
                        ~boundary:boundary_env ~side:bd.side ~unit:bd.unit
                        ~key:k c.from_imported_inputs
                    in
                    let* () =
                      require_lossless ~side:bd.side ~unit:bd.unit ~canonical:k
                        checked
                    in
                    let* expected =
                      match Key_map.find_opt k specs with
                      | Some s -> Ok s
                      | None ->
                          fail (Plan_error.make ~canonical:k Unknown_member)
                    in
                    let* () =
                      require_spec ~side:bd.side ~unit:bd.unit ~canonical:k
                        ~expected checked
                    in
                    per_read more)
          in
          let* () = per_read (reads_of_side bd.unit bd.side) in
          per_boundary rest
    in
    per_boundary (Partition_check.boundaries partition)
  in
  (* Output bindings: one writer per canonical value, none that is also
     initial; both candidate recipes lossless and well-scoped. *)
  let* writers =
    let rec go writers = function
      | [] -> Ok writers
      | (b : Output_binding.t) :: rest ->
          if not (unit_exists b.unit) then
            fail (Plan_error.make ~unit:b.unit ~canonical:b.key Unknown_member)
          else if not (Key_map.mem b.key specs) then
            fail (Plan_error.make ~unit:b.unit ~canonical:b.key Unknown_member)
          else if List.exists (Id.Canonical_value.equal b.key) initials then
            fail
              (Plan_error.make ~unit:b.unit ~canonical:b.key
                 Conflicting_initial_value)
          else if Key_map.mem b.key writers then
            fail (Plan_error.make ~unit:b.unit ~canonical:b.key Duplicate_owner)
          else go (Key_map.add b.key b.unit writers) rest
    in
    go Key_map.empty frontier.output_bindings
  in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | k :: rest ->
          if
            Key_map.mem k writers
            || List.exists (Id.Canonical_value.equal k) initials
          then go rest
          else fail (Plan_error.make ~canonical:k Missing_owner)
    in
    go all_keys
  in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (b : Output_binding.t) :: rest ->
          let check_side side recipe =
            match Partition_check.boundary partition b.unit side with
            | None -> fail (Plan_error.make ~side ~unit:b.unit Unknown_member)
            | Some bd ->
                let boundary_env =
                  tensor_env
                    (List.filter_map
                       (fun t ->
                         Option.map (fun s -> (t, s)) (tensor_spec side t))
                       bd.outputs)
                in
                let* checked =
                  match
                    check_recipe ~options
                      ~canonical:(canonical_env (reads_of b.unit))
                      ~boundary:boundary_env ~side ~unit:b.unit ~key:b.key
                      recipe
                  with
                  | Ok c -> Ok c
                  | Error e ->
                      (* An unknown name in a candidate recipe is a recipe
                         reaching outside this unit's outputs, not a missing
                         input. *)
                      let k = Err.Error.kind e in
                      let code : Failure.code =
                        match k.code with
                        | Missing_common_input -> Boundary_mismatch
                        | c -> c
                      in
                      fail { k with Plan_error.code }
                in
                let* () =
                  require_lossless ~side ~unit:b.unit ~canonical:b.key checked
                in
                let* expected =
                  match Key_map.find_opt b.key specs with
                  | Some s -> Ok s
                  | None ->
                      fail (Plan_error.make ~canonical:b.key Unknown_member)
                in
                let* () =
                  require_spec ~side ~unit:b.unit ~canonical:b.key ~expected
                    checked
                in
                Ok checked
          in
          let* l = check_side Side.Left b.from_left_outputs in
          let* r = check_side Side.Right b.from_right_outputs in
          ignore (l, r);
          go rest
    in
    go frontier.output_bindings
  in
  (* An operand another unit produces must be fed from a canonical value that
     unit writes and whose candidate on this side reads that very tensor;
     otherwise the operand would silently take an unrelated (possibly
     initial) value and the frontier would not carry the computation. *)
  let candidate_reads side writer_unit key tensor =
    List.exists
      (fun (b : Output_binding.t) ->
        Id.Unit.equal b.unit writer_unit
        && Id.Canonical_value.equal b.key key
        &&
        let recipe =
          match (side : Side.t) with
          | Left -> b.from_left_outputs
          | Right -> b.from_right_outputs
        in
        let arg = function
          | Recipe.Boundary t -> Tensor_ref.equal t tensor
          | Canonical _ | Conversion_result _ -> false
        in
        arg recipe.result
        || List.exists
             (fun (n : Recipe.node) -> List.exists arg n.arguments)
             recipe.nodes)
      frontier.output_bindings
  in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (unit, side, tensor, reads) :: rest -> (
          match Graph_index.producer (idx side) tensor with
          | Some (Graph_index.Producer.Node { node = p; _ }) -> (
              match Partition_check.owner partition side p with
              | Some a when not (Id.Unit.equal a unit) ->
                  let carried =
                    List.exists
                      (fun k ->
                        match Key_map.find_opt k writers with
                        | Some w ->
                            Id.Unit.equal w a && candidate_reads side a k tensor
                        | None -> false)
                      reads
                  in
                  if carried then go rest
                  else
                    fail
                      (Plan_error.make ~side ~unit ~tensor Missing_common_input)
              | _ -> go rest)
          | _ -> go rest)
    in
    go (List.rev !binding_reads)
  in
  let* pair_of = Pair_check.check ~options ~left ~right ~partition m in
  let all_pairs =
    List.concat_map
      (fun (r : Tensor_relation.t) -> r.member_pairs)
      m.tensor_relations
  in
  ignore all_pairs;
  (* Required outgoing values are carried by a candidate recipe; diagnostics
     need a pair, a recipe or an explicit not-compared disposition. *)
  let reads_boundary side unit tensor =
    List.exists
      (fun (b : Output_binding.t) ->
        Id.Unit.equal b.unit unit
        &&
        let recipe =
          match (side : Side.t) with
          | Left -> b.from_left_outputs
          | Right -> b.from_right_outputs
        in
        let arg = function
          | Recipe.Boundary t -> Tensor_ref.equal t tensor
          | Canonical _ | Conversion_result _ -> false
        in
        arg recipe.result
        || List.exists
             (fun (n : Recipe.node) -> List.exists arg n.arguments)
             recipe.nodes)
      frontier.output_bindings
  in
  let in_pair side unit tensor =
    match Partition_check.matched_unit partition unit with
    | None -> false
    | Some u ->
        List.exists
          (fun pid ->
            match pair_of pid with
            | None -> false
            | Some p ->
                Tensor_ref.equal tensor
                  (match (side : Side.t) with
                  | Left -> p.left
                  | Right -> p.right))
          u.output_pairs
  in
  let not_compared =
    List.filter_map
      (function
        | Member_disposition.Tensor_not_compared { tensor; reason } ->
            Some (tensor, reason)
        | _ -> None)
      m.dispositions
  in
  let* partial =
    let rec per_boundary acc = function
      | [] -> Ok (List.rev acc)
      | (bd : Partition_check.Boundary.t) :: rest ->
          let rec per_out acc = function
            | [] -> Ok acc
            | (t : Tensor_ref.t) :: more -> (
                let required =
                  List.exists (Tensor_ref.equal t) bd.required_outgoing
                in
                if reads_boundary bd.side bd.unit t then per_out acc more
                else if required then
                  fail
                    (Plan_error.make ~side:bd.side ~unit:bd.unit ~tensor:t
                       Missing_tensor_pair)
                else if in_pair bd.side bd.unit t then per_out acc more
                else
                  match
                    List.find_opt
                      (fun (x, _) -> Tensor_ref.equal x t)
                      not_compared
                  with
                  | Some (_, reason) -> per_out (reason :: acc) more
                  | None ->
                      fail
                        (Plan_error.make ~side:bd.side ~unit:bd.unit ~tensor:t
                           Missing_tensor_pair))
          in
          let* acc = per_out acc bd.outputs in
          per_boundary acc rest
    in
    per_boundary [] (Partition_check.boundaries partition)
  in
  (* Ordered final outputs. *)
  let lefts = Graph_index.final_outputs left
  and rights = Graph_index.final_outputs right in
  let* () =
    let n = List.length frontier.final_outputs in
    if n <> List.length lefts || n <> List.length rights then
      fail
        (Plan_error.make
           ~detail:
             (Expected_actual_count { expected = List.length lefts; actual = n })
           Output_order_mismatch)
    else if List.length lefts <> List.length rights then
      fail
        (Plan_error.make
           ~detail:
             (Expected_actual_count
                { expected = List.length lefts; actual = List.length rights })
           Output_order_mismatch)
    else Ok ()
  in
  let* () =
    let rec go i fs ls rs =
      match (fs, ls, rs) with
      | [], [], [] -> Ok ()
      | (f : Final_output.t) :: fs, l :: ls, r :: rs -> (
          if not (Tensor_ref.equal f.left l) then
            fail
              (Plan_error.make ~side:Left ~tensor:f.left ~detail:(Position i)
                 Output_order_mismatch)
          else if not (Tensor_ref.equal f.right r) then
            fail
              (Plan_error.make ~side:Right ~tensor:f.right ~detail:(Position i)
                 Output_order_mismatch)
          else if not (Key_map.mem f.canonical specs) then
            fail (Plan_error.make ~canonical:f.canonical Unknown_member)
          else
            match Key_map.find_opt f.canonical writers with
            | Some u ->
                if
                  reads_boundary Left u f.left && reads_boundary Right u f.right
                then go (i + 1) fs ls rs
                else
                  fail
                    (Plan_error.make ~unit:u ~canonical:f.canonical
                       ~detail:(Position i) Output_order_mismatch)
            | None ->
                (* An initial value returned unchanged: both endpoints must
                   return their source directly. *)
                if
                  List.exists (Id.Canonical_value.equal f.canonical) initials
                  && Graph_index.is_source left f.left
                  && Graph_index.is_source right f.right
                then go (i + 1) fs ls rs
                else
                  fail
                    (Plan_error.make ~canonical:f.canonical ~detail:(Position i)
                       Output_order_mismatch))
      | _ -> fail (Plan_error.make Output_order_mismatch)
    in
    go 0 frontier.final_outputs lefts rights
  in
  (* Unit plans. *)
  let plans =
    List.map
      (fun u ->
        let m_unit =
          match Partition_check.matched_unit partition u with
          | Some x -> x
          | None -> invalid_arg "Correspondence_check: unit"
        in
        let inputs_of side =
          match Partition_check.boundary partition u side with
          | Some bd -> bd.inputs
          | None -> []
        in
        let side_bindings side =
          List.filter_map
            (fun t ->
              List.find_opt
                (fun (b : Input_binding.t) ->
                  Id.Unit.equal b.unit u && Side.equal b.side side
                  && Tensor_ref.equal b.tensor t)
                frontier.input_bindings)
            (inputs_of side)
        in
        let checks =
          List.filter
            (fun (c : Input_check.t) -> Id.Unit.equal c.unit u)
            frontier.input_checks
          |> List.sort (fun (a : Input_check.t) (b : Input_check.t) ->
              match Side.compare a.side b.side with
              | 0 -> Id.Canonical_value.compare a.canonical b.canonical
              | c -> c)
        in
        let outs =
          List.filter
            (fun (b : Output_binding.t) -> Id.Unit.equal b.unit u)
            frontier.output_bindings
          |> List.sort (fun (a : Output_binding.t) (b : Output_binding.t) ->
              Id.Canonical_value.compare a.key b.key)
        in
        {
          Unit_plan.unit = u;
          canonical_reads = reads_of u;
          canonical_writes = List.map (fun (b : Output_binding.t) -> b.key) outs;
          input_bindings = side_bindings Side.Left @ side_bindings Side.Right;
          input_checks = checks;
          output_bindings = outs;
          output_pairs = List.filter_map pair_of m_unit.output_pairs;
        })
      (Partition_check.units partition)
  in
  let coverage =
    match partial with
    | [] -> Coverage.Complete
    | reasons -> Coverage.Partial reasons
  in
  Ok
    {
      plans;
      writers;
      initials;
      specs;
      finals =
        List.map
          (fun (f : Final_output.t) -> f.canonical)
          frontier.final_outputs;
      coverage;
    }

let unit_plans t = t.plans

let unit_plan t u =
  List.find_opt (fun (p : Unit_plan.t) -> Id.Unit.equal p.unit u) t.plans

let writer t k = Key_map.find_opt k t.writers
let initial t k = List.exists (Id.Canonical_value.equal k) t.initials
let canonical_spec t k = Key_map.find_opt k t.specs
let final_values t = t.finals
let coverage t = t.coverage
let _ = index_of
