(* Shared helpers for the permute-rewrite-pass tests split out of what was
   permute_passes_test.ml: [run]/[rewritten] drive a pass list
   over a fixture graph (loud and quiet forms), [evaluated]/[output_tensor]/
   [same_tensor] check the rewrite is numerically transparent, and [matches]
   inspects what a [Pattern] finds independent of what the pass then builds.
   See .ai/native_transform_design.md §12. *)

let run ?(show_before = true) g passes =
  match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error (Err.Error.kind e)
  | Ok (Rewrite.Origin state) -> (
      if show_before then
        Format.printf "@[<v 2>before:@,%a@]@." Graph_ir.pp (Rewrite.graph state);
      match Pass.run_all state passes with
      | Error e -> Format.printf "%a@." Pass.pp_error (Err.Error.kind e)
      | Ok (Rewrite.Step (final, map)) ->
          Format.printf "@[<v 2>after:@,%a@]@." Graph_ir.pp
            (Rewrite.graph final);
          Format.printf "@[<v 2>map:@,%a@]@." Graph_map.pp map)

(* Quiet form, for the numeric checks where the graphs are a means and the
   values are the point.

   The error payload is deliberately discarded: every caller feeds the result
   into a boolean equivalence check, so a failure here surfaces as that check
   reporting [false] against a golden that expects [true]. The test still fails,
   loudly enough — turning these into [Err.or_raise ~pp_error:] would restructure every
   caller's [Some]/[None] match for a diagnostic they do not read. *)
let rewritten g passes =
  match Rewrite.origin g with
  | Error _ -> None
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state passes with
      | Error _ -> None
      | Ok (Rewrite.Step (final, _)) -> Some (Rewrite.graph final))

(* [inputs] pairs positionally with [g.Graph_ir.Graph.inputs], so a fixture
   with N declared inputs needs N tensors here — [reshape_to_permute]'s numeric
   test passes one, [sink_permute_broadcast]'s passes two independently
   shaped ones. *)
let evaluated g inputs =
  match
    Eval_direct.run g ~inputs:(List.combine g.Graph_ir.Graph.inputs inputs)
  with
  | Error e -> Format.asprintf "%a" Eval_direct.pp_error (Err.Error.kind e)
  | Ok env -> (
      match g.Graph_ir.Graph.outputs with
      | [ out ] -> Format.asprintf "%a" Tensor.pp (Tensor_id.Map.find out env)
      | _ -> "expected exactly one output")

(* The single-output tensor itself, for a comparison that isn't limited to
   [Tensor.pp]'s abbreviated first-8-elements printout. Discards the evaluation
   error for the same reason as [rewritten] above. *)
let output_tensor g inputs =
  match
    Eval_direct.run g ~inputs:(List.combine g.Graph_ir.Graph.inputs inputs)
  with
  | Error _ -> None
  | Ok env -> (
      match g.Graph_ir.Graph.outputs with
      | [ out ] -> Some (Tensor_id.Map.find out env)
      | _ -> None)

(* Every coordinate, not just [Tensor.pp]'s truncated preview. *)
let same_tensor (Tensor.Tensor a) (Tensor.Tensor b) =
  Stdlib.( = ) a.Tensor.shape b.Tensor.shape
  &&
  let same = ref true in
  Vec6.iter a.Tensor.shape (fun c ->
      if
        not
          (Float.equal
             (Tensor.read (Tensor.Tensor a) c)
             (Tensor.read (Tensor.Tensor b) c))
      then same := false);
  !same

(* What a pass matches, independent of what it then builds. *)
let matches pattern g =
  match Graph_view.of_graph g with
  | Error e -> Format.printf "view: %a@." Graph_view.pp_error (Err.Error.kind e)
  | Ok view ->
      let found = Pattern.scan pattern view in
      if found = [] then Format.printf "no match@."
      else
        List.iter
          (fun (_, region) ->
            Format.printf "@[<v 2>matched %a:@,%a@]@." Region.pp region
              Graph_ir.pp
              (Region.extract view region))
          found
