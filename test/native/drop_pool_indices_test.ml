(* Narrowing [Max_pool2d_with_indices] to [Max_pool2d] once its index output is
   dead. See .ai/native_transform_design.md §12g.

   The point of this file is the CLAIM, not the rewrite. The pass asserts the
   pooled value is [Identical] across the narrowing, and on a real model the
   verifier cannot check that — resnet18's pooled tensor is 56x56x64, two hundred
   thousand coordinates, so every cluster comes back `unproved (too large)` and
   the cram golden records a budget decline rather than a proof. A claim that is
   only ever declined is a claim nobody has checked, so it is proved here instead,
   on a tensor small enough to exhaust. *)

let build name m =
  Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m
  |> Core.or_raise (fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

let pool_params : Pool.MaxPool2d.params =
  {
    kernel = { h = Dim.extent 2; w = Dim.extent 2 };
    stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
    pad = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
  }

(* 4x4x2 in, so 2x2x2 = 8 pooled coordinates: inside every budget, so the
   verdict is a real exhaustive proof rather than a decline. The index edge is
   simply never used — after [Dce] has taken the [Discard] sink away, that is
   exactly the shape the pass sees. *)
let unused_indices () =
  build "unused_indices"
    (let open Graph_builder in
     let* x = input ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:2) () in
     let* values, _indices = max_pool2d_with_indices pool_params x in
     relu values)

(* A live index edge, to show the pass declines rather than narrowing something
   it would break. *)
let live_indices () =
  build "live_indices"
    (let open Graph_builder in
     let* x = input ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:2) () in
     let* values, indices = max_pool2d_with_indices pool_params x in
     add values indices)

let%expect_test "drop_pool_indices: the narrowing, and its map" =
  let g = unused_indices () in
  (match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state [ Drop_pool_indices.pass ] with
      | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
      | Ok (Rewrite.Step (final, map)) ->
          Format.printf "@[<v 2>after:@,%a@]@." Graph_ir.pp
            (Rewrite.graph final);
          Format.printf "@[<v 2>map:@,%a@]@." Graph_map.pp map));
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [H=4 W=4 C=2] ->[n2]]
      nodes:
        n2: [t1 f32 [H=2 W=2 C=2] ->[n1]] =
          max_pool2d
            x=t0
            params={kernel={h=2; w=2}; stride={h=2; w=2}; pad={h=0; w=0}}
        n1: [t3 f32 [H=2 W=2 C=2]] = relu x=t1 <-n2
      outputs: [t3 f32 [H=2 W=2 C=2] <-n1]
    map:
      values:
        {t2} -> {} identical
      nodes:
        {n0} -> {n2}
      provenance:
        none |}]

let%expect_test "drop_pool_indices: a live index edge is left alone" =
  let g = live_indices () in
  (match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state [ Drop_pool_indices.pass ] with
      | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
      | Ok (Rewrite.Step (_, map)) ->
          Format.printf "changed: %b@."
            (not
               (Correspondence.is_empty (Graph_map.values map)
               && Node_map.is_empty (Graph_map.nodes map)))));
  [%expect {| changed: false |}]

(* THE CLAIM. [Require_proved] is the development bar — an unproved rewrite is a
   rewrite nobody has justified — so this failing would mean the [Identical]
   label is asserting more than the two compute paths deliver. They are in fact
   the same call: [MaxPool2dWithIndices.Compute.value_pixel] and
   [MaxPool2d.Compute.pixel] both reduce to [S.max_pool2d] with the same
   parameters, so the verifier compares two structurally equal terms. *)
let%expect_test "drop_pool_indices: the Identical claim is proved, not declined"
    =
  let g = unused_indices () in
  match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind
  | Ok (Rewrite.Origin state) ->
      (match
         Pass.run_reporting ~verify:Map_verify.Policy.Require_proved state
           [ Drop_pool_indices.pass ]
       with
      | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
      | Ok { Pass.audits; _ } ->
          List.iter
            (fun { Pass.Audit.pass; report } ->
              Format.printf "%s: %s@." pass (Map_verify.Report.summary report))
            audits);
      [%expect
        {| drop_pool_indices: 4 clusters: 3 proved (structural), 1 vacuous |}]
