(* Shared helpers for the transform-verifier tests split out of what was
   verify_test.ml. The symbolic transformation verifier: run a
   pass, then check the map it produced actually holds — without payloads for
   the graph inputs, so a [proved] verdict is a statement about EVERY input
   rather than one sample. See .ai/native_transform_verify.md.

   Goldens carry verdicts and ids only, never magnitudes: a printed float would
   make them depend on the platform's floating point, a verdict does not (the
   same reason fold_batch_norm_test.ml prints one).

   [error]/[pp_error]/[pp_result]/[lift_origin]/[lift_pass]/[lift_verify],
   [s]/[build]/[verify_map]/[hand_map], [relu_of], [verified_with]/
   [check_with], and [hw_ramp] are used across more than one split file;
   anything narrower (e.g. [boundary_of]/[policy_row] in verify_boundary_test.ml,
   [cell]/[show_norm] in verify_rounding_test.ml, [w_shape] in
   verify_constants_test.ml, [payloads]/[run_with_payloads] in
   verify_obligations_test.ml) stays local to the one file that needs it. *)

type error = [ Map_verify.error | `Origin of Rewrite.error | Pass.error ]

let pp_error ppf : [< error ] -> unit = function
  | #Map_verify.error as e -> Map_verify.pp_error ppf e
  | `Origin e -> Rewrite.pp_error ppf e
  | #Pass.error as e -> Pass.pp_error ppf e

let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:pp_error

let lift_origin (r : ('a, Rewrite.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> `Origin e) r

let lift_pass (r : ('a, Pass.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> (e :> error)) r

let lift_verify (r : ('a, Map_verify.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> (e :> error)) r

(* Run [passes] over [g] and verify the map that comes back. *)
let verified g passes =
  let open Err.Syntax in
  let* (Rewrite.Origin state) = lift_origin (Rewrite.origin g) in
  let* step = lift_pass (Pass.run_all state passes) in
  lift_verify (Map_verify.step state step)

let check name g passes =
  Format.printf "%s: %a@." name
    (pp_result (fun ppf r ->
         Format.fprintf ppf "%s" (Map_verify.Report.summary r)))
    (verified g passes)

let detail name g passes =
  Format.printf "@[<v 2>%s:@,%a@]@." name
    (pp_result Map_verify.Report.pp_verdicts)
    (verified g passes)

let s = Graph_fixtures.nhwc ~h:3 ~w:3 ~c:2

let build name m =
  Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let verify_map map ~src ~dst =
  Format.printf "%a@."
    (pp_result Map_verify.Report.pp_verdicts)
    (lift_verify (Map_verify.run map ~src ~dst))

(* A map assembled by hand, at the two graphs' own versions. [Graph_map.create]
   is the only constructor, so a test states its clusters through the snapshots
   exactly as production code does. *)
let hand_map ~src ~dst values nodes =
  Graph_map.create ~src ~dst ~values ~nodes ~provenance:Provenance.empty
  |> Result.get_ok

let relu_of op () =
  build "g"
    Graph_builder.(
      let* a = input ~shape:s ~name:"a" () in
      let* b = input ~shape:s ~name:"b" () in
      let* c = op a b in
      relu c)

let verified_with ?budget ?probe ~constants g passes =
  let open Err.Syntax in
  let* (Rewrite.Origin state) = lift_origin (Rewrite.origin ~constants g) in
  let* step = lift_pass (Pass.run_all state passes) in
  lift_verify (Map_verify.step ?budget ?probe state step)

let check_with ?budget ?probe ~constants name g passes =
  Format.printf "@[<v 2>%s:@,%a@]@." name
    (pp_result Map_verify.Report.pp_verdicts)
    (verified_with ?budget ?probe ~constants g passes)

let hw_ramp shape =
  Tensor.materialize shape (fun c ->
      float_of_int
        ((Dim.to_int (Vec6.get c Axis.H) * 10) + Dim.to_int (Vec6.get c Axis.W)))
