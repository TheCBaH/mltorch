(* See err_host.mli. *)

let trace_var = "MLTORCH_ERROR_TRACE"
let backtrace_var = "MLTORCH_ERROR_BACKTRACE"
let max_events_var = "MLTORCH_ERROR_MAX_EVENTS"
let max_frames_var = "MLTORCH_ERROR_MAX_FRAMES"
let max_external_bytes_var = "MLTORCH_ERROR_MAX_EXTERNAL_BYTES"

let vars =
  [
    backtrace_var;
    max_events_var;
    max_external_bytes_var;
    max_frames_var;
    trace_var;
  ]

let pp_error = Err.Config.pp_of_strings_error

let parse lookup =
  let trace = lookup trace_var
  and backtrace = lookup backtrace_var
  and max_events = lookup max_events_var
  and max_frames = lookup max_frames_var
  and max_external_bytes = lookup max_external_bytes_var in
  (* All five absent means "leave the policy alone". Not the same as passing
     five [None]s to [of_strings], which would return -- and therefore install
     -- [Err]'s defaults, silently overriding a policy the host had already
     chosen for itself. *)
  if
    List.for_all Option.is_none
      [ backtrace; max_events; max_external_bytes; max_frames; trace ]
  then Err.return None
  else
    Err.map
      (fun config -> Some config)
      (Err.Config.of_strings ~trace ~backtrace ~max_events ~max_frames
         ~max_external_bytes)

let install_from_env () =
  let open Err.Syntax in
  let+ config = parse Sys.getenv_opt in
  Option.iter Err.Config.set config;
  config
