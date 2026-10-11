open Inference_compare_schema
module Plan = Inference_compare_plan
module P = Plan_fixtures
module U = Plan.Unit_order

let m = P.mapping
let left = P.left_graph
let right = P.right_graph
let l = P.left_key
let r = P.right_key
let t = P.tensor
let uid = P.uid

let%expect_test "the reference plan" =
  (match P.plan m with
  | Error e -> Format.printf "ERROR %a@." P.pp_error e
  | Ok p ->
      Format.printf "order: %a@."
        Fmt.(list ~sep:(any " -> ") Id.Unit.pp)
        (U.order (Plan.Structural.order p));
      List.iter
        (fun (u : Plan.Correspondence_check.Unit_plan.t) ->
          Format.printf "%a reads=[%a] writes=[%a] inputs=%d checks=%d@."
            Id.Unit.pp u.unit
            Fmt.(list ~sep:comma Id.Canonical_value.pp)
            u.canonical_reads
            Fmt.(list ~sep:comma Id.Canonical_value.pp)
            u.canonical_writes
            (List.length u.input_bindings)
            (List.length u.input_checks))
        (Plan.Structural.units p);
      Format.printf "coverage: %s@."
        (match Plan.Structural.coverage p with
        | Complete -> "complete"
        | Partial _ -> "partial");
      Format.printf "finals: %a@."
        Fmt.(list ~sep:comma Id.Canonical_value.pp)
        (Plan.Structural.final_values p);
      List.iter
        (fun (lt : U.Lifetime.t) ->
          Format.printf "%a: writer=%s reads=%d final=%d total=%d@."
            Id.Canonical_value.pp lt.canonical
            (match lt.writer with
            | None -> "initial"
            | Some u -> Id.Unit.to_string u)
            (List.length lt.reads) lt.final_uses lt.total_uses)
        (U.lifetimes (Plan.Structural.order p));
      List.iter
        (fun (e : U.Edge.t) ->
          Format.printf "edge %a -> %a (%s)@." Id.Unit.pp e.from_unit
            Id.Unit.pp e.to_unit
            (match e.reason with
            | Canonical c -> "canonical " ^ Id.Canonical_value.to_string c
            | Data { side; _ } -> "data " ^ Side.to_string side
            | Control { side; _ } -> "control " ^ Side.to_string side))
        (U.edges (Plan.Structural.order p)));
  [%expect {|
    order: u1 -> u2
    u1 reads=[c1,
    x] writes=[h] inputs=4 checks=4
    u2 reads=[c2,
    h] writes=[y] inputs=4 checks=4
    coverage: complete
    finals: y
    c1: writer=initial reads=1 final=0 total=4
    c2: writer=initial reads=1 final=0 total=4
    h: writer=u1 reads=1 final=0 total=4
    x: writer=initial reads=1 final=0 total=4
    y: writer=u2 reads=0 final=1 total=1
    edge u1 -> u2 (canonical h)
    edge u1 -> u2 (data left)
    edge u1 -> u2 (data right) |}]

let%expect_test "set order does not change the plan" =
  let shuffled : Mapping_artifact.t =
    {
      m with
      units = List.rev m.units;
      dispositions = List.rev m.dispositions;
      frontier =
        {
          m.frontier with
          input_bindings = List.rev m.frontier.input_bindings;
          input_checks = List.rev m.frontier.input_checks;
          output_bindings = List.rev m.frontier.output_bindings;
          canonical_values = List.rev m.frontier.canonical_values;
          initial_values = List.rev m.frontier.initial_values;
        };
    }
  in
  let reversed_graph (g : Graph_desc.t) =
    { g with nodes = List.rev g.nodes; sources = List.rev g.sources }
  in
  (match
     ( P.plan m,
       P.plan ~left:(reversed_graph left) ~right:(reversed_graph right) shuffled )
   with
  | Ok a, Ok b ->
      Format.printf "same order: %b@."
        (U.order (Plan.Structural.order a) = U.order (Plan.Structural.order b));
      Format.printf "same mapping preimage: %b@."
        (String.equal
           (Plan.Structural.mapping_preimage a)
           (Plan.Structural.mapping_preimage b));
      Format.printf "same unit plans: %b@."
        (List.map
           (fun (u : Plan.Correspondence_check.Unit_plan.t) ->
             (u.unit, u.canonical_reads, u.canonical_writes))
           (Plan.Structural.units a)
        = List.map
            (fun (u : Plan.Correspondence_check.Unit_plan.t) ->
              (u.unit, u.canonical_reads, u.canonical_writes))
            (Plan.Structural.units b))
  | Error e, _ | _, Error e -> Format.printf "ERROR %a@." P.pp_error e);
  [%expect {|
    same order: true
    same mapping preimage: true
    same unit plans: true |}]

let with_units f : Mapping_artifact.t =
  { m with units = List.map f m.units }

let with_frontier f : Mapping_artifact.t = { m with frontier = f m.frontier }

let%expect_test "stale and unknown identities" =
  let other_digest =
    {
      left with
      key = { left.key with graph_digest = P.(digest_error (Digest.Graph.make sha256 ~hex:(String.make 64 'f'))) };
    }
  in
  P.outcome ~left:other_digest "stale left graph digest" m;
  P.outcome "mapping built for another right graph"
    { m with right_graph = { r with version = P.(text (Id.Graph_version.of_string "v2")) } };
  P.outcome "unknown node in a cluster"
    (with_units (fun u ->
         if Id.Unit.to_string u.unit_id = "u1" then
           { u with left = { u.left with nodes = P.node l "ghost" :: u.left.nodes } }
         else u));
  P.outcome "cluster from the other graph"
    (with_units (fun u ->
         if Id.Unit.to_string u.unit_id = "u1" then
           { u with left = { u.left with nodes = P.node r "m2" :: u.left.nodes } }
         else u));
  [%expect {|
    stale left graph digest: stale_graph side=left
    mapping built for another right graph: stale_graph side=right
    unknown node in a cluster: unknown_member side=left unit=u1 node=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/ghost
    cluster from the other graph: stale_graph side=left unit=u1 node=native/v1@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:/m2 |}]

let%expect_test "partition completeness" =
  P.outcome "node owned twice"
    (with_units (fun u ->
         if Id.Unit.to_string u.unit_id = "u2" then
           { u with left = { u.left with nodes = P.node l "n1" :: u.left.nodes } }
         else u));
  P.outcome "helper omitted from every unit"
    (with_units (fun u ->
         if Id.Unit.to_string u.unit_id = "u1" then
           { u with right = { u.right with nodes = [ P.node r "m1b" ] } }
         else u));
  let with_assertion =
    let assertion =
      P.node_desc ~effect:Metadata_assertion l "assert_shape" "assert" [ "x" ] []
    in
    { left with nodes = left.nodes @ [ assertion ] }
  in
  P.outcome ~left:with_assertion "assertion owned by no unit" m;
  P.outcome ~left:with_assertion "assertion as its own zero-output unit"
    {
      m with
      units =
        {
          unit_id = uid "u_assert";
          left = P.cluster l [ "assert_shape" ] [ "x" ] [];
          right = P.cluster r [] [] [];
          output_pairs = [];
        }
        :: m.units;
      frontier =
        {
          m.frontier with
          input_bindings =
            P.binding "u_assert" Left (t l "x")
              (P.identity_from_canonical "ba" "x" P.m23)
            :: m.frontier.input_bindings;
          input_checks =
            P.check "u_assert" Left "x"
              (P.identity_from_boundary "ka" (t l "x") P.m23)
            :: m.frontier.input_checks;
        };
    };
  P.outcome "helper disposition naming the wrong owner"
    { m with dispositions = [ Internal_helper { node = P.node r "m1a"; owner_unit = uid "u2" } ] };
  [%expect {|
    node owned twice: duplicate_owner side=left unit=u2 node=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/n1
    helper omitted from every unit: missing_owner side=right node=native/v1@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:/m1a
    assertion owned by no unit: missing_owner side=left node=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/assert_shape
    assertion as its own zero-output unit: ok units=[u1 u2 u_assert]
    helper disposition naming the wrong owner: boundary_mismatch side=right unit=u2 node=native/v1@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:/m1a |}]

let%expect_test "unit boundaries are recomputed" =
  let edit_u1 f =
    with_units (fun u ->
        if Id.Unit.to_string u.unit_id = "u1" then f u else u)
  in
  P.outcome "an external operand left out of the inputs"
    (edit_u1 (fun u ->
         { u with left = { u.left with inputs = [ t l "x" ] } }));
  P.outcome "an input nothing inside reads"
    (edit_u1 (fun u ->
         { u with left = { u.left with inputs = t l "c2" :: u.left.inputs } }));
  P.outcome "an outgoing value left out"
    (edit_u1 (fun u -> { u with left = { u.left with outputs = [] } }));
  P.outcome "an output that is neither produced inside nor passed through"
    (edit_u1 (fun u ->
         { u with left = { u.left with outputs = t l "y" :: u.left.outputs } }));
  P.outcome "an output listed twice"
    (edit_u1 (fun u ->
         { u with left = { u.left with outputs = [ t l "h"; t l "h" ] } }));
  [%expect {|
    an external operand left out of the inputs: boundary_mismatch side=left unit=u1 tensor=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/c1
    an input nothing inside reads: boundary_mismatch side=left unit=u1 tensor=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/c2
    an outgoing value left out: boundary_mismatch side=left unit=u1 tensor=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/h
    an output that is neither produced inside nor passed through: boundary_mismatch side=left unit=u1 tensor=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/y
    an output listed twice: duplicate_owner side=left unit=u1 tensor=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/h |}]

let%expect_test "correspondence and frontier recipes" =
  let drop_binding f = with_frontier (fun fr ->
      { fr with input_bindings = List.filter (fun b -> not (f b)) fr.input_bindings })
  in
  P.outcome "residual/constant operand without a binding"
    (drop_binding (fun (b : Input_binding.t) ->
         Tensor_ref.equal b.tensor (t l "c2")));
  P.outcome "a second binding for one operand"
    (with_frontier (fun fr ->
         {
           fr with
           input_bindings =
             P.binding "u2" Left (t l "c2") (P.identity_from_canonical "dup" "c2" P.m23)
             :: fr.input_bindings;
         }));
  P.outcome "missing reconstruction check"
    (with_frontier (fun fr ->
         {
           fr with
           input_checks =
             List.filter
               (fun (c : Input_check.t) ->
                 not
                   (Id.Unit.to_string c.unit = "u2"
                   && c.side = Right
                   && Id.Canonical_value.to_string c.canonical = "h"))
               fr.input_checks;
         }));
  P.outcome "a binding whose spec is not the operand's"
    (with_frontier (fun fr ->
         {
           fr with
           input_bindings =
             List.map
               (fun (b : Input_binding.t) ->
                 if Tensor_ref.equal b.tensor (t r "xt") then
                   { b with from_common = P.identity_from_canonical "wrong" "x" P.m23 }
                 else b)
               fr.input_bindings;
         }));
  let narrowing =
    let n = P.node_id "cast" in
    P.recipe ~lossless:true "narrow"
      [
        {
          id = n;
          operation =
            Cast
              {
                target =
                  {
                    codec = Inference_compare_codecs.Element_codec.i32;
                    parameters_digest = None;
                  };
                rounding = Nearest_even;
                overflow = Reject;
              };
          arguments = [ Recipe.Canonical (P.ckey "c1") ];
        };
      ]
      (Recipe.Conversion_result n)
      { P.m23 with dtype = { codec = Inference_compare_codecs.Element_codec.i32; parameters_digest = None } }
  in
  ignore narrowing;
  P.outcome "both candidates required: right recipe reads a left tensor"
    (with_frontier (fun fr ->
         {
           fr with
           output_bindings =
             List.map
               (fun (b : Output_binding.t) ->
                 if Id.Unit.to_string b.unit = "u1" then
                   { b with from_right_outputs = P.identity_from_boundary "swap" (t l "h") P.m23 }
                 else b)
               fr.output_bindings;
         }));
  P.outcome "writer removed: nothing owns the value"
    (with_frontier (fun fr ->
         {
           fr with
           output_bindings =
             List.filter
               (fun (b : Output_binding.t) -> Id.Unit.to_string b.unit <> "u1")
               fr.output_bindings;
         }));
  (* The right unit exposes a diagnostic xr and the candidate recipe reads
     it, so the outgoing hr that u2 consumes is carried by no recipe. *)
  P.outcome "required outgoing value carried by no candidate"
    {
      (with_units (fun u ->
           if Id.Unit.to_string u.unit_id = "u1" then
             { u with right = { u.right with outputs = [ t r "xr"; t r "hr" ] } }
           else u))
      with
      frontier =
        {
          m.frontier with
          output_bindings =
            List.map
              (fun (b : Output_binding.t) ->
                if Id.Unit.to_string b.unit = "u1" then
                  { b with from_right_outputs = P.identity_from_boundary "diag" (t r "xr") P.m23 }
                else b)
              m.frontier.output_bindings;
        };
    };
  P.outcome "canonical value both initial and written"
    (with_frontier (fun fr ->
         { fr with initial_values = P.payload "h" '9' :: fr.initial_values }));
  P.outcome "canonical value with no owner"
    (with_frontier (fun fr ->
         {
           fr with
           canonical_values = { key = P.ckey "z"; spec = P.m23 } :: fr.canonical_values;
         }));
  P.outcome "two writers for one value"
    (with_frontier (fun fr ->
         {
           fr with
           output_bindings =
             {
               (List.hd fr.output_bindings) with
               unit = uid "u2";
               comparison_id = P.comparison "cmp_dup";
             }
             :: fr.output_bindings;
         }));
  [%expect {|
    residual/constant operand without a binding: missing_common_input side=left unit=u2 tensor=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/c2
    a second binding for one operand: ambiguous_route side=left unit=u2 tensor=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/c2
    missing reconstruction check: missing_common_input side=right unit=u2 canonical=h
    a binding whose spec is not the operand's: boundary_mismatch side=right unit=u1 tensor=native/v1@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:/xt spec_mismatch
    both candidates required: right recipe reads a left tensor: boundary_mismatch side=right unit=u1 canonical=h recipe_fault=unknown boundary tensor aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/h
    writer removed: nothing owns the value: missing_owner canonical=h
    required outgoing value carried by no candidate: missing_common_input side=right unit=u2 tensor=native/v1@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:/hr
    canonical value both initial and written: conflicting_initial_value unit=u1 canonical=h
    canonical value with no owner: missing_owner canonical=z
    two writers for one value: duplicate_owner unit=u1 canonical=h |}]

let%expect_test "ordered final outputs" =
  P.outcome "wrong right tensor at an occurrence"
    (with_frontier (fun fr ->
         {
           fr with
           final_outputs =
             List.map
               (fun (f : Final_output.t) -> { f with right = t r "hr" })
               fr.final_outputs;
         }));
  P.outcome "occurrence omitted"
    (with_frontier (fun fr -> { fr with final_outputs = [] }));
  (* Two outputs, [y] then [h], and the swapped listing. *)
  let two_left = { left with final_outputs = [ t l "y"; t l "h" ];
                  output_tree = Tuple [ Tensor (t l "y"); Tensor (t l "h") ] }
  and two_right = { right with final_outputs = [ t r "yr"; t r "hr" ];
                   output_tree = Tuple [ Tensor (t r "yr"); Tensor (t r "hr") ] } in
  let occ s = P.text (Id.Output_occurrence.of_string s) in
  let two key_a key_b =
    with_frontier (fun fr ->
        {
          fr with
          final_outputs =
            [
              { occurrence = occ "o0"; canonical = P.ckey "y"; left = t l key_a; right = t r (if key_a = "y" then "yr" else "hr") };
              { occurrence = occ "o1"; canonical = P.ckey "h"; left = t l key_b; right = t r (if key_b = "y" then "yr" else "hr") };
            ];
        })
  in
  P.outcome ~left:two_left ~right:two_right "two outputs in order" (two "y" "h");
  P.outcome ~left:two_left ~right:two_right "two outputs swapped" (two "h" "y");
  (* The same tensor returned twice keeps both occurrences. *)
  let dup_left = { left with final_outputs = [ t l "y"; t l "y" ];
                  output_tree = Tuple [ Tensor (t l "y"); Tensor (t l "y") ] }
  and dup_right = { right with final_outputs = [ t r "yr"; t r "yr" ];
                   output_tree = Tuple [ Tensor (t r "yr"); Tensor (t r "yr") ] } in
  P.outcome ~left:dup_left ~right:dup_right "repeated output, one occurrence listed"
    m;
  P.outcome ~left:dup_left ~right:dup_right "repeated output, both listed"
    (with_frontier (fun fr ->
         {
           fr with
           final_outputs =
             [
               { occurrence = occ "o0"; canonical = P.ckey "y"; left = t l "y"; right = t r "yr" };
               { occurrence = occ "o1"; canonical = P.ckey "y"; left = t l "y"; right = t r "yr" };
             ];
         }));
  [%expect {|
    wrong right tensor at an occurrence: output_order_mismatch side=right tensor=native/v1@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:/hr position=0
    occurrence omitted: output_order_mismatch expected=1 actual=0
    two outputs in order: ok units=[u1 u2]
    two outputs swapped: output_order_mismatch side=left tensor=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/h position=0
    repeated output, one occurrence listed: output_order_mismatch expected=2 actual=1
    repeated output, both listed: ok units=[u1 u2] |}]

let%expect_test "union dependency cycle" =
  P.outcome ~left:P.cyc_left ~right:P.cyc_right "opposing endpoint dependencies"
    P.cyc_mapping;
  (match
     Plan.Structural.check ~limits:P.limits ~options:P.options ~left:P.cyc_left
       ~right:P.cyc_right P.cyc_mapping
   with
  | Error e -> (
      match (Err.Error.kind e).detail with
      | Cycle edges ->
          List.iter
            (fun (c : Plan.Plan_error.Cycle_edge.t) ->
              Format.printf "  %a -> %a: %s@." Id.Unit.pp c.from_unit
                Id.Unit.pp c.to_unit
                (match c.reason with
                | Data { side; tensor } ->
                    Printf.sprintf "data %s %s" (Side.to_string side)
                      (Id.Tensor_local.to_string tensor.local_id)
                | Control { side; _ } -> "control " ^ Side.to_string side
                | Canonical k -> "canonical " ^ Id.Canonical_value.to_string k))
            edges
      | _ -> Format.printf "no witness@.")
  | Ok _ -> Format.printf "accepted@.");
  [%expect {|
    opposing endpoint dependencies: unit_cycle unit=A cycle_length=4
      A -> B: canonical p
      A -> B: data left p
      B -> A: canonical q
      B -> A: data right r |}]

let%expect_test "endpoint graph validity and limits" =
  let bad_operand =
    {
      left with
      nodes =
        P.node_desc l "n1" "add" [ "x"; "nowhere" ] [ "h" ] :: List.tl left.nodes;
    }
  in
  P.outcome ~left:bad_operand "operand with no producer" m;
  let unsupported =
    {
      left with
      nodes =
        List.map
          (fun (n : Node_desc.t) ->
            if Id.Node_local.to_string n.node.local_id = "n2" then
              {
                n with
                effect =
                  Unsupported (P.text (Prose.Description.of_string "rng"));
              }
            else n)
          left.nodes;
    }
  in
  P.outcome ~left:unsupported "unsupported effect" m;
  let self_loop =
    {
      left with
      nodes =
        List.map
          (fun (n : Node_desc.t) ->
            if Id.Node_local.to_string n.node.local_id = "n1" then
              { n with control_predecessors = [ P.node l "n2" ] }
            else n)
          left.nodes;
    }
  in
  P.outcome ~left:self_loop "cycle inside one endpoint" m;
  let tree_mismatch =
    { left with output_tree = Tuple [ Tensor (t l "h") ] }
  in
  P.outcome ~left:tree_mismatch "output tree disagrees with the outputs" m;
  P.outcome ~limits:{ P.limits with max_units = 1L } "unit count over the limit" m;
  P.outcome ~limits:{ P.limits with max_nodes = 2L } "node count over the limit" m;
  [%expect {|
    operand with no producer: unknown_member side=left node=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/n1 tensor=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/nowhere
    unsupported effect: unsupported_effect side=left node=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/n2
    cycle inside one endpoint: unit_cycle side=left node_cycle_length=2
    output tree disagrees with the outputs: output_order_mismatch side=left position=0
    unit count over the limit: resource_exhausted limit=1 actual=2
    node count over the limit: resource_exhausted side=right limit=2 actual=3 |}]

let%expect_test "an operand must be fed by the unit that produces it" =
  (* u2's left operand h is produced by u1; binding it to an initial value
     would let the engine run on a stale tensor. *)
  P.outcome "operand bound to an unrelated initial value"
    (with_frontier (fun fr ->
         {
           fr with
           input_bindings =
             List.map
               (fun (b : Input_binding.t) ->
                 if Id.Unit.to_string b.unit = "u2" && b.side = Left
                    && Tensor_ref.equal b.tensor (t l "h")
                 then
                   { b with from_common = P.identity_from_canonical "stale" "c1" P.m23 }
                 else b)
               fr.input_bindings;
           input_checks =
             List.map
               (fun (c : Input_check.t) ->
                 if Id.Unit.to_string c.unit = "u2" && c.side = Left
                    && Id.Canonical_value.to_string c.canonical = "h"
                 then
                   P.check "u2" Left "c1"
                     (P.identity_from_boundary "kc1" (t l "h") P.m23)
                 else c)
               fr.input_checks;
         }));
  [%expect {| operand bound to an unrelated initial value: missing_common_input side=left unit=u2 tensor=aten/v1@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:/h |}]

let%expect_test "a lossy common-frontier binding is refused even when declared lossy" =
  let i32 =
    { P.m23 with dtype = { codec = Inference_compare_codecs.Element_codec.i32; parameters_digest = None } }
  in
  let n = P.node_id "cast" in
  let narrowing =
    P.recipe ~lossless:false "lossy"
      [
        {
          id = n;
          operation =
            Cast
              {
                target = i32.dtype;
                rounding = Nearest_even;
                overflow = Saturate;
              };
          arguments = [ Recipe.Canonical (P.ckey "c1") ];
        };
      ]
      (Recipe.Conversion_result n) i32
  in
  P.outcome "float constant fed through an integer cast"
    (with_frontier (fun fr ->
         {
           fr with
           input_bindings =
             List.map
               (fun (b : Input_binding.t) ->
                 if Tensor_ref.equal b.tensor (t r "k1") then
                   { b with from_common = narrowing }
                 else b)
               fr.input_bindings;
         }));
  [%expect {| float constant fed through an integer cast: lossy_common_input side=right unit=u1 tensor=native/v1@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb:/k1 recipe_fault=recipe declared lossless but is lossy |}]
