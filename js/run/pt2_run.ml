(* The pure-engine counterpart of test/interp_run.ml: the same all-sample top-5
   report, evaluated by [Native_interp] instead of by ATen. Built natively here
   and as a node program in js/jsoo.
   argv: <model.pt2> <images_dir> <synsets.txt> <metadata.txt> <results.json>
         [--cram] [--strict]

   NOT a probe, which is why it lives here rather than in js/probe. The probe
   dumps every output bit of ONE image so the Makefile can diff the two backends
   against each other -- it answers "do native and node agree". This answers a
   different question, "is the answer right", by comparing every sample's top-5
   against the rankings shipped in the release. Neither subsumes the other: two
   backends can agree on a wrong answer, and a right answer says nothing about
   whether they agree bit for bit.

   [Sys.time] rather than [Unix.gettimeofday]: [unix] must not enter the
   js_of_ocaml closure, and the timings only go to stderr. *)

(* [Native_interp] runs the graph; [Native_predict] turns its packed outputs into
   a ranking. Two libraries, so the classifier tail is testable without a model
   -- see test/native_interp/predict_test.ml. *)
type eval = [ Native_interp.error | Native_predict.error ]

let pp_eval ppf : eval -> unit = function
  | #Native_predict.error as e -> Native_predict.pp_error ppf e
  | #Native_interp.error as e -> Native_interp.pp_error ppf e

let infer archive image =
  let open Err.Syntax in
  let* outputs =
    Native_interp.run archive ~input:image
    |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))
  in
  Native_predict.top_predictions outputs 5
  |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))

let () =
  match Infer_report.parse_argv Sys.argv with
  | Error usage ->
      prerr_endline usage;
      exit 2
  | Ok (paths, options) -> (
      match Infer_report.run ~now:Sys.time ~infer paths options with
      | Ok () -> ()
      | Error e ->
          Format.eprintf "%a@." (Err.Error.pp (Infer_report.pp_error pp_eval)) e;
          exit 1)
