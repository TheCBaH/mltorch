(* The whole-model verification this plan exists for: a real downloaded
   model's Region-authored nodes run through Loop_js_exec-compiled
   JavaScript, in-process, under node, checked against the release's own
   top-5 rankings. No native counterpart -- js_of_ocaml-only by
   construction, since [Loop_region_executor] is (see
   js/jsoo/loop_js_exec_js/dune). MANUAL, like `jsoo.pt2.run`: a single
   fastvit_sa12 sample measured at ~103s natively makes even one sample,
   let alone the full ten `results.json` holds, unsuitable for a blocking
   CI step.

   argv: <model.pt2> <inputs.pt> <expected.json> <outputs.pt> [--cram]
         [--strict]
   Same four positional paths as js/run/pt2_run.ml -- deliberately runs only
   the FIRST of the (many) samples they hold ([~max_samples:(Some 1)]),
   never the whole map: see [Infer_report.report]'s own doc for why that
   parameter exists. *)

type eval = [ Native_interp.error | Native_predict.error ]

let pp_eval ppf : eval -> unit = function
  | #Native_predict.error as e -> Native_predict.pp_error ppf e
  | #Native_interp.error as e -> Native_interp.pp_error ppf e

let coverage = Loop_region_executor.Coverage.create ()
let region_executor = Loop_region_executor.make coverage

let infer archive image =
  let open Err.Syntax in
  let* outputs =
    Native_interp.run ~region_executor archive ~input:image
    |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))
  in
  let* top =
    Native_predict.top_predictions outputs 5
    |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))
  in
  Err.return (List.map (fun ((c : Dim.index Dim.t), p) -> ((c :> int), p)) top)

(* T3.3: a run that silently skipped the generated-JS path is a worse
   defect than one that fails loudly, since nothing else about a passing
   ranking check would tell them apart. Checked BEFORE the ranking result
   is even inspected, so a coverage failure is never masked by (or
   mistaken for) a ranking one. *)
let min_generated_js = 1

let () =
  match Infer_report.parse_argv Sys.argv with
  | Error usage ->
      prerr_endline usage;
      exit 2
  | Ok (paths, options) -> (
      let report_result =
        Infer_report.run ~max_samples:1 ~now:Sys.time ~infer paths options
      in
      match Loop_region_executor.Coverage.check ~min_generated_js coverage with
      | Error msg ->
          Format.eprintf "loop_js_pt2: %s@." msg;
          exit 1
      | Ok () -> (
          match report_result with
          | Ok () -> ()
          | Error e ->
              Format.eprintf "%a@."
                (Err.Error.pp (Infer_report.pp_error pp_eval))
                e;
              exit 1))
