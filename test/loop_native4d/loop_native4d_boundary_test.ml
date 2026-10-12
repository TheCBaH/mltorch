open Loop_ir
module Common = Loop_native4d_common

(* A budget a Region computation cannot meet: [max_local_slots] bounds the
   trace/scalar/vector storage of a Region's locals. Limits may only tighten. *)
let tight =
  let d = Kernel.Limits.default in
  Kernel.Limits.create ~max_size:d.max_size ~max_depth:d.max_depth
    ~max_values:d.max_values ~max_dep_depth:d.max_dep_depth
    ~max_inputs:d.max_inputs ~max_outputs:d.max_outputs ~max_extent:d.max_extent
    ~max_numel:d.max_numel ~max_bytes:d.max_bytes ~max_local_slots:1
    ~max_scan_state:d.max_scan_state
    ~max_scan_updates_per_key:d.max_scan_updates_per_key
    ~max_scan_updates_total:d.max_scan_updates_total
  |> Err.or_raise ~pp_error:Kernel.Limits.pp_error

let softmax = Native4d_test.Fixtures.softmax_over Axis.W ()

let tag_native : Loop_bundle.error -> string = function
  | `Region_construction _ -> "Region_construction"
  | `Missing_tensor _ -> "Missing_tensor"
  | `Missing_tensor_sig _ -> "Missing_tensor_sig"
  | _ -> "other"

let tag_generic : _ Loop_bundle.generic_error -> string = function
  | `Dialect _ -> "Dialect"
  | `Missing_signature _ -> "Missing_signature"
  | `Region_construction _ -> "Region_construction"
  | _ -> "other"

(* The budget is refused by the Region validation the BUILDER runs on what a
   dialect returns, so it surfaces as the shared arm in every dialect, and the
   Native wrapper reports it as the arm it always did. *)
let%expect_test "a Region over budget: shared arm, Native row unchanged" =
  (match Err.payload (Loop_bundle.build ~limits:tight softmax) with
  | Ok _ -> print_endline "native build: accepted"
  | Error e ->
      Fmt.pr "native build: %s: %a@." (tag_native e) Loop_bundle.pp_error e);
  (match
     Err.payload
       (Loop_bundle.build_with ~dialect:Loop_dialect.native ~limits:tight
          softmax)
   with
  | Ok _ -> print_endline "native build_with: accepted"
  | Error e -> Fmt.pr "native build_with: %s@." (tag_generic e));
  let c = Common.lowered_case "softmax_w" softmax in
  (match Err.payload (Loop_native4d.build ~limits:tight c.Common.graph) with
  | Ok _ -> print_endline "native4d build: accepted"
  | Error e ->
      Fmt.pr "native4d build: %s: %a@." (tag_generic e) Loop_native4d.pp_error e);
  [%expect
    {|
    native build: Region_construction: total local slot count exceeds limit 1
    native build_with: Region_construction
    native4d build: Region_construction: total local slot count exceeds limit 1 |}]

(* A graph whose signatures disagree with its edges is the dialect's own
   failure, behind [`Dialect]: the storage script is the first reader. *)
let%expect_test "a missing signature is the dialect's own error" =
  let c = Common.lowered_case "softmax_w" softmax in
  let g = c.Common.graph in
  let out = List.hd g.Graph_common.Graph.outputs in
  let broken =
    {
      g with
      Graph_common.Graph.tensors =
        Tensor_id.Map.remove out g.Graph_common.Graph.tensors;
    }
  in
  (match Err.payload (Loop_native4d.build broken) with
  | Ok _ -> print_endline "accepted"
  | Error e -> Fmt.pr "%s: %a@." (tag_generic e) Loop_native4d.pp_error e);
  [%expect {| Dialect: missing tensor sig t1 |}]

(* The grouped path ([Region_execution.lower_group]) reports through the same
   shared arm. *)
let%expect_test "a Region group over budget: shared arm in both dialects" =
  let lstm = Native4d_test.Fixtures.lstm_states_live () in
  (match Err.payload (Loop_bundle.build ~limits:tight lstm) with
  | Ok _ -> print_endline "native build: accepted"
  | Error e ->
      Fmt.pr "native build: %s: %a@." (tag_native e) Loop_bundle.pp_error e);
  let c = Common.lowered_case "lstm_states_live" lstm in
  (match Err.payload (Loop_native4d.build ~limits:tight c.Common.graph) with
  | Ok _ -> print_endline "native4d build: accepted"
  | Error e ->
      Fmt.pr "native4d build: %s: %a@." (tag_generic e) Loop_native4d.pp_error e);
  [%expect
    {|
    native build: Region_construction: total local slot count exceeds limit 1
    native4d build: Region_construction: total local slot count exceeds limit 1 |}]
