open Loop_ir
open Graph_ir

(* W2 acceptance (plan): schedule/admission reuse, Region refusal named by
   node, dead-index suppression via the shared [Eval_direct.dry_run] fold.
   [chain]/[sink_permute_layer_norm] are [Native_test.Graph_fixtures]' own
   shared fixtures, not new graphs invented for this test. *)

let pp_ids ppf ids =
  Fmt.pf ppf "[%a]" Fmt.(list ~sep:comma (using Tensor_id.to_int int)) ids

let%expect_test "Node-path graph: one invocation per node, in schedule order" =
  let g = Native_test.Graph_fixtures.chain () in
  (match Err.payload (Loop_bundle.build g) with
  | Error e -> Fmt.pr "build failed: %a@." Loop_bundle.pp_error e
  | Ok b ->
      Fmt.pr "inputs=%a@." pp_ids b.Loop_bundle.inputs;
      Fmt.pr "constants=%a@." pp_ids b.Loop_bundle.constants;
      Fmt.pr "outputs=%a@." pp_ids b.Loop_bundle.outputs;
      Fmt.pr "invocation count=%d@." (List.length b.Loop_bundle.invocations);
      List.iter
        (fun (inv : Loop_bundle.invocation) ->
          let o = List.hd inv.Loop_bundle.outputs in
          Fmt.pr "node=%a output=%d oid=t%d role=%a arena=%a buffers=%a@."
            Node_id.pp inv.Loop_bundle.node
            (Output_ordinal.to_int o.Loop_bundle.Output.ordinal)
            (Tensor_id.to_int o.Loop_bundle.Output.oid)
            Storage_script.Role.pp o.Loop_bundle.Output.role
            Fmt.(option Storage_script.Arena_id.pp)
            o.Loop_bundle.Output.arena
            Fmt.(list ~sep:comma string)
            (List.map
               (fun (buf : Loop_buffer.t) ->
                 Printf.sprintf "%s:t%d"
                   (Loop_buffer.role_name buf.role)
                   (Tensor_id.to_int buf.id))
               inv.Loop_bundle.program.Loop_program.buffers))
        b.Loop_bundle.invocations;
      Fmt.pr "plan arenas=%a@."
        Fmt.(list ~sep:comma Storage_script.Arena_id.pp)
        (List.map fst (Storage_plan.arenas b.Loop_bundle.plan)));
  [%expect
    {|
    inputs=[0]
    constants=[1, 2, 3, 4, 5,
    6]
    outputs=[9]
    invocation count=3
    node=n0 output=0 oid=t7 role=intermediate arena=intermediates buffers=in:t0,
    in:t1, in:t2,
    out:t7
    node=n1 output=0 oid=t8 role=intermediate arena=intermediates buffers=in:t7,
    in:t3, in:t4, in:t5, in:t6,
    out:t8
    node=n2 output=0 oid=t9 role=output arena=outputs buffers=in:t8,
    out:t9
    plan arenas=constants, inputs, intermediates,
    outputs |}]

(* W2 acceptance (plan): "both layouts" and "borrowed bindings" -- the same
   [chain] fixture through [Storage_script.Config.t] corners other than
   [Loop_bundle.default_config] ([Separate]/[Copied]/[Copied]), to confirm the
   plumbing [Loop_bundle.build] already exposes via [?config] actually reaches
   [Storage_plan]/[Arena_plan] correctly for each -- not merely that a caller
   COULD pass them. *)
let print_summary b =
  Fmt.pr "invocation count=%d@." (List.length b.Loop_bundle.invocations);
  List.iter
    (fun (inv : Loop_bundle.invocation) ->
      let o = List.hd inv.Loop_bundle.outputs in
      Fmt.pr "node=%a role=%a arena=%a@." Node_id.pp inv.Loop_bundle.node
        Storage_script.Role.pp o.Loop_bundle.Output.role
        Fmt.(option Storage_script.Arena_id.pp)
        o.Loop_bundle.Output.arena)
    b.Loop_bundle.invocations;
  Fmt.pr "plan arenas=%a@."
    Fmt.(list ~sep:comma Storage_script.Arena_id.pp)
    (List.map fst (Storage_plan.arenas b.Loop_bundle.plan))

let%expect_test
    "Shared_execution layout: intermediates/outputs share one arena, constants \
     separate" =
  let g = Native_test.Graph_fixtures.chain () in
  let config : Storage_script.Config.t =
    {
      layout = Storage_script.Layout.Shared_execution;
      constants = Storage_script.Ownership.Copied;
      inputs = Storage_script.Ownership.Copied;
    }
  in
  (match Err.payload (Loop_bundle.build ~config g) with
  | Error e -> Fmt.pr "build failed: %a@." Loop_bundle.pp_error e
  | Ok b -> print_summary b);
  [%expect
    {|
    invocation count=3
    node=n0 role=intermediate arena=execution
    node=n1 role=intermediate arena=execution
    node=n2 role=output arena=execution
    plan arenas=constants,
    execution |}]

let%expect_test "Borrowed constants and inputs: no arena, outside every pool" =
  let g = Native_test.Graph_fixtures.chain () in
  let config : Storage_script.Config.t =
    {
      layout = Storage_script.Layout.Separate;
      constants = Storage_script.Ownership.Borrowed;
      inputs = Storage_script.Ownership.Borrowed;
    }
  in
  (match Err.payload (Loop_bundle.build ~config g) with
  | Error e -> Fmt.pr "build failed: %a@." Loop_bundle.pp_error e
  | Ok b ->
      print_summary b;
      Fmt.pr "inputs=%a constants=%a@." pp_ids b.Loop_bundle.inputs pp_ids
        b.Loop_bundle.constants);
  [%expect
    {|
    invocation count=3
    node=n0 role=intermediate arena=intermediates
    node=n1 role=intermediate arena=intermediates
    node=n2 role=output arena=outputs
    plan arenas=intermediates,
    outputs
    inputs=[0] constants=[1, 2, 3, 4, 5,
    6] |}]

(* W2 acceptance (plan): "forwarded/repeated outputs" -- a graph output that
   IS a graph input (never computed by any node) and a graph output list that
   names the same edge twice. Built directly with [Graph_builder] rather than
   a shared fixture: no existing [Native_test.Graph_fixtures] graph forwards
   an input, and this shape is used only here. *)
let forwarded_input_graph () =
  Graph_builder.build ~name:"forwarded_input"
    ~outputs:(fun (x, o) -> [ x; o ])
    Graph_builder.(
      let* x = input ~shape:(Native_test.Graph_fixtures.s1c 4) () in
      let+ o = relu x in
      (x, o))
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "fixture forwarded_input: %a" Graph_builder.pp_error e)

let%expect_test
    "forwarded input: graph output that is a graph input, never produced by \
     any invocation" =
  let g = forwarded_input_graph () in
  (match Err.payload (Loop_bundle.build g) with
  | Error e -> Fmt.pr "build failed: %a@." Loop_bundle.pp_error e
  | Ok b ->
      Fmt.pr "inputs=%a outputs=%a@." pp_ids b.Loop_bundle.inputs pp_ids
        b.Loop_bundle.outputs;
      Fmt.pr "invocation count=%d@." (List.length b.Loop_bundle.invocations);
      Fmt.pr "forwarded edge arena=%a@."
        Fmt.(option (option Storage_script.Arena_id.pp))
        (Storage_script.events b.Loop_bundle.script
        |> List.find_map (function
          | Storage_script.Event.Alloc
              { Storage_script.Block.alloc; arena; role = Input } ->
              if
                List.mem alloc.Alloc_script.Alloc.id b.Loop_bundle.inputs
                && List.mem alloc.Alloc_script.Alloc.id b.Loop_bundle.outputs
              then Some arena
              else None
          | _ -> None)));
  [%expect
    {|
    inputs=[0] outputs=[0,
    1]
    invocation count=1
    forwarded edge arena=outputs |}]

let%expect_test "repeated output: the same edge named twice in graph.outputs" =
  let g =
    Graph_builder.build ~name:"repeated_output"
      ~outputs:(fun o -> [ o; o ])
      Graph_builder.(
        let* x = input ~shape:(Native_test.Graph_fixtures.s1c 4) () in
        relu x)
    |> Err.or_raise ~pp_error:(fun ppf e ->
        Fmt.pf ppf "fixture repeated_output: %a" Graph_builder.pp_error e)
  in
  (match Err.payload (Loop_bundle.build g) with
  | Error e -> Fmt.pr "build failed: %a@." Loop_bundle.pp_error e
  | Ok b ->
      Fmt.pr "outputs=%a invocation count=%d@." pp_ids b.Loop_bundle.outputs
        (List.length b.Loop_bundle.invocations));
  [%expect {|
    outputs=[1,
    1] invocation count=1 |}]

(* A Region-authored node is an invocation like any other, built from
   signatures alone. [edges] binds each positional buffer to a graph edge (a
   Region program's own ids are local to it) and [synthetics] names the
   defaults the graph does not supply (LayerNorm's omitted weight/bias, Sdpa's
   omitted mask). *)
let%expect_test "Region-authored nodes: edges and synthetic defaults" =
  List.iter
    (fun (name, g) ->
      Fmt.pr "%s@." name;
      match Err.payload (Loop_bundle.build g) with
      | Error e -> Fmt.pr "build failed: %a@." Loop_bundle.pp_error e
      | Ok b ->
          List.iter
            (fun (inv : Loop_bundle.invocation) ->
              Fmt.pr "node=%a oid=t%d edges=%a synthetics=%a@." Node_id.pp
                inv.Loop_bundle.node
                (Tensor_id.to_int
                   (List.hd inv.Loop_bundle.outputs).Loop_bundle.Output.oid)
                pp_ids inv.Loop_bundle.edges
                Fmt.(
                  list ~sep:comma (fun ppf (s : Loop_bundle.synthetic) ->
                      pf ppf "t%d=%g"
                        (Tensor_id.to_int s.Loop_bundle.id)
                        s.Loop_bundle.value))
                inv.Loop_bundle.synthetics)
            b.Loop_bundle.invocations)
    [
      ("layer_norm", Native_test.Graph_fixtures.sink_permute_layer_norm ());
      ("sdpa", Native_test.Graph_fixtures.sink_permute_sdpa ());
    ];
  [%expect
    {|
    layer_norm
    node=n0 oid=t1 edges=[0,
    1] synthetics=
    node=n1 oid=t2 edges=[1, 3, 4, 2] synthetics=t3=0,
    t4=1
    sdpa
    node=n0 oid=t1 edges=[0,
    1] synthetics=
    node=n1 oid=t4 edges=[1, 2, 3, 8,
    4] synthetics=t8=0 |}]

let build_js g =
  match Err.payload (Loop_bundle.build g) with
  | Error e -> Fmt.pr "descriptor build failed: %a@." Loop_bundle.pp_error e
  | Ok b -> (
      match Err.payload (Loop_bundle_js.build b) with
      | Error e -> Fmt.pr "js build failed: %a@." Loop_bundle_js.pp_error e
      | Ok js ->
          Fmt.pr "invocations=%d distinct_kernels=%d pools=%d@."
            (List.length b.Loop_bundle.invocations)
            js.Loop_bundle_js.distinct_kernels
            (List.length js.Loop_bundle_js.pools);
          print_string (Js_print.script js.Loop_bundle_js.program))

let%expect_test "chain: three distinct ops intern to three distinct kernels" =
  build_js (Native_test.Graph_fixtures.chain ());
  [%expect
    {|
    invocations=3 distinct_kernels=3 pools=4
    "use strict";
    function kernel_0(b0, b1, b2, b3) {
      let x0 = 0;
      let x1 = 0;
      let x2 = 0;
      for (let i0 = 0; i0 < 3; i0++) {
        for (let i1 = 0; i1 < 3; i1++) {
          for (let i2 = 0; i2 < 3; i2++) {
            x0 = 0;
            for (let i3 = 0; i3 < 2; i3++) {
              x1 = 0;
              for (let i4 = 0; i4 < 2; i4++) {
                x2 = 0;
                for (let i5 = 0; i5 < 2; i5++) {
                  x2 = x2 + b0[2 * (4 * (i0 + i4) + (i1 + i5)) + i3] * b1[2 * (2 * (2 * i2 + i4) + i5) + i3];
                }
                x1 = x1 + x2;
              }
              x0 = x0 + x1;
            }
            b3[3 * (3 * i0 + i1) + i2] = x0 + b2[i2];
          }
        }
      }
      return null;
    }
    function kernel_1(b0, b1, b2, b3, b4, b5) {
      for (let i0 = 0; i0 < 9; i0++) {
        for (let i1 = 0; i1 < 3; i1++) {
          b5[3 * i0 + i1] = (b0[3 * i0 + i1] - b3[i1]) * (1 / Math.sqrt(b4[i1] + 1.0000000000000001e-05)) * b1[i1] + b2[i1];
        }
      }
      return null;
    }
    function kernel_2(b0, b1) {
      let x0 = 0;
      for (let i0 = 0; i0 < 27; i0++) {
        x0 = b0[i0];
        b1[i0] = x0 < 0 ? 0 : x0;
      }
      return null;
    }
    function run_bundle(pool_constants_float32, pool_inputs_float32, pool_intermediates_float32, pool_outputs_float32) {
      const e0 = pool_inputs_float32.subarray(0, 32);
      const e1 = pool_constants_float32.subarray(64, 88);
      const e2 = pool_constants_float32.subarray(96, 99);
      const e3 = pool_constants_float32.subarray(48, 51);
      const e4 = pool_constants_float32.subarray(16, 19);
      const e5 = pool_constants_float32.subarray(0, 3);
      const e6 = pool_constants_float32.subarray(32, 35);
      const e7 = pool_intermediates_float32.subarray(0, 27);
      e7.fill(0);
      const r0 = kernel_0(e0, e1, e2, e7);
      if (r0 !== null) {
        return [0, r0];
      }
      const e8 = pool_intermediates_float32.subarray(32, 59);
      e8.fill(0);
      const r1 = kernel_1(e7, e3, e4, e5, e6, e8);
      if (r1 !== null) {
        return [1, r1];
      }
      const e9 = pool_outputs_float32.subarray(0, 27);
      e9.fill(0);
      const r2 = kernel_2(e8, e9);
      if (r2 !== null) {
        return [2, r2];
      }
      return null;
    }
    |}]

(* [residual]'s two [Relu] nodes share one shape ([s1c 4]) and so print
   IDENTICAL kernel source (buffer names are positional, never global ids):
   proof that interning shares them, not just a claim about the mechanism. *)
let%expect_test "residual: two same-shape Relu invocations intern to one kernel"
    =
  build_js (Native_test.Graph_fixtures.residual ());
  [%expect
    {|
    invocations=3 distinct_kernels=2 pools=3
    "use strict";
    function kernel_0(b0, b1) {
      let x0 = 0;
      for (let i0 = 0; i0 < 4; i0++) {
        x0 = b0[i0];
        b1[i0] = x0 < 0 ? 0 : x0;
      }
      return null;
    }
    function kernel_1(b0, b1, b2) {
      for (let i0 = 0; i0 < 4; i0++) {
        b2[i0] = b0[i0] + b1[i0];
      }
      return null;
    }
    function run_bundle(pool_inputs_float32, pool_intermediates_float32, pool_outputs_float32) {
      const e0 = pool_inputs_float32.subarray(0, 4);
      const e1 = pool_intermediates_float32.subarray(0, 4);
      e1.fill(0);
      const r0 = kernel_0(e0, e1);
      if (r0 !== null) {
        return [0, r0];
      }
      const e2 = pool_intermediates_float32.subarray(16, 20);
      e2.fill(0);
      const r1 = kernel_0(e1, e2);
      if (r1 !== null) {
        return [1, r1];
      }
      const e3 = pool_outputs_float32.subarray(0, 4);
      e3.fill(0);
      const r2 = kernel_1(e0, e2, e3);
      if (r2 !== null) {
        return [2, r2];
      }
      return null;
    } |}]

(* A supplied plan is accepted only for the script it was witnessed against. *)
let%expect_test "supplied plan: accepted when equal, refused on any difference"
    =
  let g = Native_test.Graph_fixtures.chain () in
  let base =
    Err.or_raise ~pp_error:Loop_bundle.pp_error (Loop_bundle.build g)
  in
  let plan = base.Loop_bundle.plan in
  let show label r =
    match Err.payload r with
    | Ok _ -> Fmt.pr "%s: accepted@." label
    | Error e -> Fmt.pr "%s: %a@." label Loop_bundle.pp_error e
  in
  show "same config" (Loop_bundle.build ~plan g);
  show "shared layout"
    (Loop_bundle.build ~plan
       ~config:
         {
           Storage_script.Config.layout = Storage_script.Layout.Shared_execution;
           constants = Storage_script.Ownership.Copied;
           inputs = Storage_script.Ownership.Copied;
         }
       g);
  show "borrowed inputs"
    (Loop_bundle.build ~plan
       ~config:
         {
           Storage_script.Config.layout = Storage_script.Layout.Separate;
           constants = Storage_script.Ownership.Copied;
           inputs = Storage_script.Ownership.Borrowed;
         }
       g);
  show "different graph"
    (Loop_bundle.build ~plan (Native_test.Graph_fixtures.residual ()));
  [%expect
    {|
    same config: accepted
    shared layout: the supplied storage plan's script differs from this graph's at @0
    borrowed inputs: the supplied storage plan's script differs from this graph's at @0
    different graph: the supplied storage plan's script differs from this graph's at @1 |}]

(* Quantized storage lives outside every arena and no lowered kernel stores it:
   the bundle refuses the graph when it is built, naming the edge, never at run
   time and never by aliasing it into a pool. *)
let%expect_test "quantized edges are refused at preparation" =
  let shape = Vec6.shape ~n:2 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3 in
  let q = Quant.per_tensor ~scale:0.25 ~zero_point:2 in
  let g =
    Native_test.Graph_fixtures.buildn "quantized_unbind"
      Graph_builder.(
        let* xq = input ~shape ~fmt:(Payload.Fmt Payload.I8) ~quant:q () in
        let* f = input ~shape:(Native_test.Graph_fixtures.s1c 4) () in
        let* slices = unbind { Split.Unbind.axis = Axis.N } xq in
        let* a = relu f in
        return (slices @ [ a ]))
  in
  (match Err.payload (Loop_bundle.build g) with
  | Error e -> Fmt.pr "bundle: %a@." Loop_bundle.pp_error e
  | Ok b -> (
      Fmt.pr "bundle built, %d invocations@."
        (List.length b.Loop_bundle.invocations);
      match Err.payload (Loop_bundle_js.build b) with
      | Error e -> Fmt.pr "js: %a@." Loop_bundle_js.pp_error e
      | Ok _ -> Fmt.pr "js built@."));
  [%expect
    {| bundle: t2: a stored value must be f32 or bool and unquantized, got i8 |}]
