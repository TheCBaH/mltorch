open Inference_compare_schema
open Util
open Components
open Units
open Routes
module R = Recipe_ops

let build ctx ~(first : Mapping_artifact.t) ~(second : Mapping_artifact.t)
    ~units ~fwd ~rev =
  (* frontier *)
  let ff = first.frontier and sf = second.frontier in
  let first_ids cu = unit_ids cu.comp.first
  and second_ids cu = unit_ids cu.comp.second in
  let left_binding_where ok a =
    find_first
      (fun (b : Input_binding.t) ->
        if b.side = Left && Tensor_ref.equal b.tensor a && ok b.unit then Some b
        else None)
      ff.input_bindings
  in
  (* A binding of [a] from the unit's own constituents, else from any unit:
     what it computes depends on [a] alone. *)
  let left_binding cu a =
    match left_binding_where (fun u -> has_unit u (first_ids cu)) a with
    | Some b -> Some b
    | None -> left_binding_where (fun _ -> true) a
  in
  let right_binding cu c =
    find_first
      (fun (b : Input_binding.t) ->
        if
          b.side = Right
          && Tensor_ref.equal b.tensor c
          && has_unit b.unit (second_ids cu)
        then Some b
        else None)
      sf.input_bindings
  in
  let reid cu tag (r : Recipe.t) text =
    let* id =
      recipe_id ctx (Id.Unit.to_string cu.id ^ "|" ^ tag ^ "|" ^ text)
    in
    Ok { r with Recipe.id }
  in
  (* [r] reads A tensors; read the same value from the C tensors instead, by
     the reverse route of each. *)
  let through_reverse (r : Recipe.t) ~c_set =
    let rec go (r : Recipe.t) = function
      | [] -> Ok r
      | a :: rest -> (
          match
            List.find_opt
              (fun d ->
                Tensor_ref.equal d.destination a && mem_tensor d.source c_set)
              rev
          with
          | None -> fail (E.Missing_route { tensor = a; toward = Right })
          | Some d ->
              let inner =
                R.chain ~id:r.id ~argument:(Recipe.Boundary d.source)
                  ~steps:d.steps ~output_spec:r.output_spec
              in
              go (R.inline ~outer:r ~argument:(Recipe.Boundary a) ~inner) rest)
    in
    go r (R.boundaries r)
  in
  (* Whether the later map relates [c] to a tensor the first map reaches from a
     tensor it binds: then a missing route is a gap, not a constant. A tensor
     related only to ones the first map never binds (a helper edge, a folded
     value) is the later map's own to pin. *)
  let has_counterpart c =
    let reached m =
      List.exists
        (fun (r : Tensor_relation.t) ->
          mem_tensor m r.right_members
          && List.exists
               (fun l -> left_binding_where (fun _ -> true) l <> None)
               r.left_members)
        first.tensor_relations
    in
    List.exists
      (fun (r : Tensor_relation.t) ->
        mem_tensor c r.right_members && List.exists reached r.left_members)
      second.tensor_relations
  in
  let constants = Hashtbl.create 8 in
  let* per_unit =
    all
      (fun cu ->
        let* left_in =
          all
            (fun a ->
              match left_binding cu a with
              | None -> fail (E.Missing_binding a)
              | Some b ->
                  let* r = reid cu "bl" b.from_common (tkey a) in
                  Ok { b with Input_binding.unit = cu.id; from_common = r })
            cu.left.inputs
        in
        let unit_constants = ref [] in
        let* right_in =
          all
            (fun c ->
              match right_binding cu c with
              | None -> fail (E.Missing_binding c)
              | Some rb -> (
                  (* Only a route from a tensor the first map binds can carry
                     a value: one from an internal tensor (a folded value, a
                     helper) is no route at all. *)
                  let to_c =
                    List.filter
                      (fun d ->
                        Tensor_ref.equal d.destination c
                        && left_binding cu d.source <> None)
                      fwd
                  in
                  match
                    match
                      List.find_opt
                        (fun d -> mem_tensor d.source cu.left.inputs)
                        to_c
                    with
                    | Some d -> Some d
                    | None -> List.nth_opt to_c 0
                  with
                  | Some d -> (
                      match left_binding cu d.source with
                      | None -> fail (E.Missing_binding d.source)
                      | Some lb ->
                          let outer =
                            R.chain ~id:rb.from_common.id
                              ~argument:(Recipe.Boundary d.source)
                              ~steps:d.steps
                              ~output_spec:rb.from_common.output_spec
                          in
                          let r =
                            R.inline ~outer ~argument:(Recipe.Boundary d.source)
                              ~inner:lb.from_common
                          in
                          let* r = reid cu "br" r (tkey c) in
                          Ok
                            {
                              rb with
                              Input_binding.unit = cu.id;
                              from_common = r;
                            })
                  | None when has_counterpart c ->
                      (* the later map relates [c] to a tensor of the shared
                         graph, but no route reaches it: not a constant *)
                      fail (E.Missing_route { tensor = c; toward = Left })
                  | None ->
                      (* a terminal constant the later map pins itself *)
                      let keys =
                        List.filter_map
                          (function Recipe.Canonical k -> Some k | _ -> None)
                          (rb.from_common.result
                          :: List.concat_map
                               (fun (n : Recipe.node) -> n.arguments)
                               rb.from_common.nodes)
                      in
                      let pinned k =
                        List.find_opt
                          (fun (i : Initial_binding.t) ->
                            Id.Canonical_value.equal i.key k)
                          sf.initial_values
                      in
                      if
                        R.boundaries rb.from_common <> []
                        || keys = []
                        || List.exists (fun k -> pinned k = None) keys
                      then fail (E.Missing_route { tensor = c; toward = Left })
                      else (
                        List.iter
                          (fun k ->
                            unit_constants := k :: !unit_constants;
                            Hashtbl.replace constants
                              (Id.Canonical_value.to_string k)
                              k)
                          keys;
                        let* r = reid cu "br" rb.from_common (tkey c) in
                        Ok
                          {
                            rb with
                            Input_binding.unit = cu.id;
                            from_common = r;
                          })))
            cu.right.inputs
        in
        (* Every canonical value a binding reads is checked on that side: the
           left check comes from the first map, the right one by reading the
           same A tensors through the reverse routes. *)
        let keys_of bindings =
          List.fold_left
            (fun acc (b : Input_binding.t) ->
              let r = b.from_common in
              List.fold_left
                (fun acc -> function
                  | Recipe.Canonical k
                    when not (List.exists (Id.Canonical_value.equal k) acc) ->
                      acc @ [ k ]
                  | _ -> acc)
                acc
                (r.result
                :: List.concat_map
                     (fun (n : Recipe.node) -> n.arguments)
                     r.nodes))
            [] bindings
        in
        let pinned k =
          List.exists
            (fun (i : Initial_binding.t) -> Id.Canonical_value.equal i.key k)
            sf.initial_values
          && not
               (List.exists
                  (fun (c : Canonical_spec.t) ->
                    Id.Canonical_value.equal c.key k)
                  ff.canonical_values)
        in
        let left_check k =
          let candidates ok =
            find_first
              (fun (c : Input_check.t) ->
                if
                  c.side = Left
                  && Id.Canonical_value.equal c.canonical k
                  && ok c
                then Some c
                else None)
              ff.input_checks
          in
          match candidates (fun c -> has_unit c.unit (first_ids cu)) with
          | Some c -> Some c
          | None -> candidates (fun _ -> true)
        in
        let side_checks side keys =
          all
            (fun k ->
              match left_check k with
              | None -> fail (E.Missing_check k)
              | Some c ->
                  let key = Id.Canonical_value.to_string k in
                  let* recipe =
                    match (side : Side.t) with
                    | Left -> Ok c.from_imported_inputs
                    | Right ->
                        through_reverse c.from_imported_inputs
                          ~c_set:cu.right.inputs
                  in
                  let* r = reid cu (Side.to_string side) recipe key in
                  Ok
                    {
                      c with
                      Input_check.unit = cu.id;
                      side;
                      from_imported_inputs = r;
                    })
            keys
        in
        let* left_checks = side_checks Left (keys_of left_in) in
        let* right_checks =
          side_checks Right
            (List.filter (fun k -> not (pinned k)) (keys_of right_in))
        in
        let checks = [ left_checks @ right_checks ] in
        (* A constant pinned by the second map is verified there, on the right
           only. *)
        let* const_checks =
          all
            (fun k ->
              match
                find_first
                  (fun (c : Input_check.t) ->
                    if
                      c.side = Right
                      && Id.Canonical_value.equal c.canonical k
                      && has_unit c.unit (second_ids cu)
                    then Some c
                    else None)
                  sf.input_checks
              with
              | None -> fail (E.Missing_check k)
              | Some c ->
                  let* r =
                    reid cu "kc" c.from_imported_inputs
                      (Id.Canonical_value.to_string k)
                  in
                  Ok
                    {
                      c with
                      Input_check.unit = cu.id;
                      from_imported_inputs = r;
                    })
            (List.sort_uniq Id.Canonical_value.compare !unit_constants)
        in
        let outputs =
          List.filter
            (fun (o : Output_binding.t) ->
              has_unit o.unit (first_ids cu)
              && R.boundaries o.from_left_outputs <> []
              && List.for_all
                   (fun t -> mem_tensor t cu.left.outputs)
                   (R.boundaries o.from_left_outputs))
            ff.output_bindings
        in
        let* () =
          match
            List.find_opt
              (fun a ->
                not
                  (List.exists
                     (fun (o : Output_binding.t) ->
                       mem_tensor a (R.boundaries o.from_left_outputs))
                     outputs))
              cu.left.outputs
          with
          | Some a -> fail (E.Missing_binding a)
          | None -> Ok ()
        in
        let* out_b =
          all
            (fun (o : Output_binding.t) ->
              let key = Id.Canonical_value.to_string o.key in
              let* l = reid cu "ol" o.from_left_outputs key in
              let* via =
                through_reverse o.from_left_outputs ~c_set:cu.right.outputs
              in
              let* r = reid cu "or" via key in
              Ok
                {
                  o with
                  Output_binding.unit = cu.id;
                  from_left_outputs = l;
                  from_right_outputs = r;
                })
            outputs
        in
        Ok (left_in @ right_in, List.concat checks @ const_checks, out_b))
      units
  in
  let input_bindings = List.concat_map (fun (b, _, _) -> b) per_unit
  and input_checks = List.concat_map (fun (_, c, _) -> c) per_unit
  and output_bindings = List.concat_map (fun (_, _, o) -> o) per_unit in
  let* final_outputs =
    let n1 = List.length first.frontier.final_outputs
    and n2 = List.length second.frontier.final_outputs in
    if n1 <> n2 then fail (E.Output_mismatch { position = min n1 n2 })
    else
      all
        (fun (i, ((f1 : Final_output.t), (f2 : Final_output.t))) ->
          if Tensor_ref.equal f1.right f2.left then
            Ok
              {
                Final_output.occurrence = f1.occurrence;
                canonical = f1.canonical;
                left = f1.left;
                right = f2.right;
              }
          else fail (E.Output_mismatch { position = i }))
        (List.mapi
           (fun i p -> (i, p))
           (List.combine first.frontier.final_outputs
              second.frontier.final_outputs))
  in
  let* canonical_values, initial_values =
    let extra =
      Hashtbl.fold (fun _ k acc -> k :: acc) constants []
      |> List.sort Id.Canonical_value.compare
    in
    let* added =
      all
        (fun k ->
          let spec =
            List.find_opt
              (fun (c : Canonical_spec.t) -> Id.Canonical_value.equal c.key k)
              sf.canonical_values
          and init =
            List.find_opt
              (fun (i : Initial_binding.t) -> Id.Canonical_value.equal i.key k)
              sf.initial_values
          in
          match (spec, init) with
          | Some spec, Some init -> (
              let mine_spec =
                List.find_opt
                  (fun (c : Canonical_spec.t) ->
                    Id.Canonical_value.equal c.key k)
                  ff.canonical_values
              in
              match mine_spec with
              | Some m when not (Tensor_spec.equal m.spec spec.spec) ->
                  fail (E.Conflicting_constant k)
              | Some _ -> (
                  match
                    List.find_opt
                      (fun (i : Initial_binding.t) ->
                        Id.Canonical_value.equal i.key k)
                      ff.initial_values
                  with
                  | Some mine
                    when Digest.Payload.equal mine.payload_digest
                           init.payload_digest ->
                      Ok (None, None)
                  | _ -> fail (E.Conflicting_constant k))
              | None -> Ok (Some spec, Some init))
          | _ -> fail (E.Conflicting_constant k))
        extra
    in
    let referenced k =
      let same = Id.Canonical_value.equal k in
      let in_recipe (r : Recipe.t) =
        List.exists
          (function Recipe.Canonical c -> same c | _ -> false)
          (r.result
          :: List.concat_map (fun (n : Recipe.node) -> n.arguments) r.nodes)
      in
      List.exists
        (fun (b : Input_binding.t) -> in_recipe b.from_common)
        input_bindings
      || List.exists
           (fun (c : Input_check.t) ->
             same c.canonical || in_recipe c.from_imported_inputs)
           input_checks
      || List.exists (fun (o : Output_binding.t) -> same o.key) output_bindings
      || List.exists
           (fun (f : Final_output.t) -> same f.canonical)
           final_outputs
    in
    Ok
      ( List.filter
          (fun (c : Canonical_spec.t) -> referenced c.key)
          (ff.canonical_values @ List.filter_map fst added),
        List.filter
          (fun (i : Initial_binding.t) -> referenced i.key)
          (ff.initial_values @ List.filter_map snd added) )
  in
  Ok
    {
      Frontier_mapping.canonical_values;
      final_outputs;
      initial_values;
      input_bindings;
      input_checks;
      output_bindings;
    }
