(* Run a .pt2 model through the ATen interpreter on every sample image and
   compare the local top-5 against the reference results.json shipped in the
   release zip. Driven by interp_cram (see `make pt2.runtest`) and
   `make inference`.
   argv: <model.pt2> <images_dir> <synsets.txt> <metadata.txt> <results.json>
         [--cram] [--strict]

   The flow itself lives in [Infer_report], shared with js/run/pt2_run.ml. All
   that is left here is the evaluator: [Interp] reaches ATen, ctypes and the C++
   runtime, which is exactly what the pure runner exists to avoid, so the two
   entry points differ in this file and nowhere else.

   [Unix.gettimeofday] is passed in rather than called by the shared library:
   [unix] must not enter the js_of_ocaml closure, and this is the runner that
   can afford it. *)

(* [Interp.run] and [Interp.top_predictions] already share [Interp.error], so
   this needs no widening -- the whole thing is one evaluator failure. *)
let infer archive image =
  let open Err.Syntax in
  let* logits = Interp.run archive image in
  Interp.top_predictions logits 5

let () =
  match Infer_report.parse_argv Sys.argv with
  | Error usage ->
      prerr_endline usage;
      exit 2
  | Ok (paths, options) -> (
      match Infer_report.run ~now:Unix.gettimeofday ~infer paths options with
      | Ok () -> ()
      | Error e ->
          Format.eprintf "%a@."
            (Err.Error.pp (Infer_report.pp_error Interp.pp_error))
            e;
          exit 1)
