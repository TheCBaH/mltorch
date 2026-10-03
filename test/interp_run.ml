(* Run a .pt2 model through the ATen interpreter on every sample image and
   compare the local top-5 against the producer's expected.json and tensor-map
   oracle. Driven by interp_cram (see `make pt2.runtest`) and
   `make inference`.
   argv: <model.pt2> <inputs.pt> <expected.json> <outputs.pt>
         [--cram] [--strict] [--safetensors]
   With --safetensors the first path is a model directory of the submodule, its
   weights mapped from the Hub checkpoint it pins (see lib/pt2_safetensors_unix).

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

(* The one place the opener's error is rendered: the shared library cannot name
   it (it must not depend on unix), so it takes the message. *)
let open_safetensors dir =
  Pt2_safetensors_unix.open_dir dir
  |> Err.export ~pos:__POS__
  |> Result.map_error (Format.asprintf "%a" Pt2_safetensors_unix.pp_error)

let () =
  match Infer_report.parse_argv Sys.argv with
  | Error usage ->
      prerr_endline usage;
      exit 2
  | Ok (paths, options) -> (
      match
        Infer_report.run ~open_safetensors ~now:Unix.gettimeofday ~infer paths
          options
      with
      | Ok () -> ()
      | Error e ->
          Format.eprintf "%a@."
            (Err.Error.pp (Infer_report.pp_error Interp.pp_error))
            e;
          exit 1)
