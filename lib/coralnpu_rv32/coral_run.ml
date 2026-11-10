module Trap = struct
  type t = { mcause : int32; mepc : int32; mtval : int32 }
end

module Expectation = struct
  type kind = Cycles | Instructions

  type t = {
    expected : int64;
    kind : kind;
    label : string option;
    limit : int64;
  }
end

module Run_error = struct
  type t =
    | Cycle_limit
    | Guest_trap of Trap.t
    | Input_refused of Coral_runtime.Refusal.t
    | Instruction_limit
    | Malformed_output of { malformed : Coral_wire.Malformed.t; tail : string }
    | Output_limit of int64
    | Over_expectation of Expectation.t
    | Simulator_crashed of { status : string; tail : string }
    | Simulator_missing of string
    | Timeout

  let pp ppf = function
    | Cycle_limit -> Fmt.string ppf "the simulator reached its cycle limit"
    | Guest_trap { mcause; mepc; mtval } ->
        Fmt.pf ppf "guest trap: mcause %ld mepc 0x%08lx mtval 0x%08lx" mcause
          mepc mtval
    | Input_refused r ->
        Fmt.pf ppf "the guest refused its input: %s"
          (Coral_runtime.Refusal.to_string r)
    | Instruction_limit ->
        Fmt.string ppf "the simulator reached its instruction limit"
    | Malformed_output { malformed; tail } ->
        Fmt.pf ppf "malformed output (%a); simulator said: %S"
          Coral_wire.Malformed.pp malformed tail
    | Output_limit n ->
        Fmt.pf ppf "the simulator wrote more than %Ld bytes; killed" n
    | Over_expectation { expected; kind; label; limit } ->
        Fmt.pf ppf "%a passed its limit of %Ld %s (%Ld expected)"
          Fmt.(option ~none:(any "the run") string)
          label limit
          (match kind with
          | Cycles -> "cycles"
          | Instructions -> "instructions")
          expected
    | Simulator_crashed { status; tail } ->
        Fmt.pf ppf "the simulator %s with no frame; it said: %S" status tail
    | Simulator_missing path -> Fmt.pf ppf "no simulator at %s" path
    | Timeout -> Fmt.string ppf "deadline passed"
end

type outcome = {
  argv : string list;
  mcycle : int64;
  minstret : int64;
  outputs : (int * string) list;
  status : int32;
  text : string;
  wall : float;
}

let tail_bytes = 4096

let tail s =
  let n = String.length s in
  if n <= tail_bytes then s else String.sub s (n - tail_bytes) tail_bytes

let describe = function
  | Unix.WEXITED n -> Printf.sprintf "exited %d" n
  | Unix.WSIGNALED n -> Printf.sprintf "was killed by signal %d" n
  | Unix.WSTOPPED n -> Printf.sprintf "stopped on signal %d" n

(* [label] names the run in the directory, so a process listing shows which case
   a stuck simulator belongs to *)
let safe_label l =
  let l = if String.length l > 48 then String.sub l 0 48 else l in
  String.map
    (function
      | ('a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '-' | '.') as c -> c | _ -> '_')
    l

let label_part = function None -> "" | Some l -> safe_label l ^ "_"

let fresh_dir label =
  (* the process id in the name: forked runners share their random state, so
     two would otherwise draw the same name and race between its removal and
     the [mkdir] *)
  let label = label_part label in
  let d =
    Filename.temp_file
      (Printf.sprintf "mltorch_coral_%s%d_" label (Unix.getpid ()))
      ""
  in
  Sys.remove d;
  Sys.mkdir d 0o700;
  d

(* Runs [argv] in [cwd] with [input] on stdin until both output pipes close
   or [deadline] passes. The child leads its own session, so a kill reaches
   everything it started. *)
let spawn ~cwd ~deadline ~limit argv input =
  let in_r, in_w = Unix.pipe ~cloexec:true () in
  let out_r, out_w = Unix.pipe ~cloexec:true () in
  let err_r, err_w = Unix.pipe ~cloexec:true () in
  match Unix.fork () with
  | 0 -> (
      try
        ignore (Unix.setsid ());
        Unix.chdir cwd;
        Unix.dup2 ~cloexec:false in_r Unix.stdin;
        Unix.dup2 ~cloexec:false out_w Unix.stdout;
        Unix.dup2 ~cloexec:false err_w Unix.stderr;
        Unix.execv (List.hd argv) (Array.of_list argv)
      with e ->
        let msg = Printexc.to_string e ^ "\n" in
        ignore (Unix.write_substring Unix.stderr msg 0 (String.length msg));
        Unix._exit 127)
  | pid ->
      List.iter Unix.close [ in_r; out_w; err_w ];
      Unix.set_nonblock in_w;
      let out = Buffer.create 65536 and err = Buffer.create 4096 in
      let buf = Bytes.create 65536 in
      let sent = ref 0 and writing = ref true in
      let reading = ref [ out_r; err_r ] in
      let close_input () =
        if !writing then (
          writing := false;
          Unix.close in_w)
      in
      if String.length input = 0 then close_input ();
      let stop = Unix.gettimeofday () +. deadline in
      let timed_out = ref false and flooded = ref false in
      while !reading <> [] && not (!timed_out || !flooded) do
        let left = stop -. Unix.gettimeofday () in
        if left <= 0. then timed_out := true
        else
          let writers = if !writing then [ in_w ] else [] in
          match Unix.select !reading writers [] left with
          | exception Unix.Unix_error (Unix.EINTR, _, _) -> ()
          | readable, writable, _ -> (
              List.iter
                (fun fd ->
                  match Unix.read fd buf 0 (Bytes.length buf) with
                  | 0 ->
                      Unix.close fd;
                      reading := List.filter (fun f -> f != fd) !reading
                  | n ->
                      let into = if fd == out_r then out else err in
                      Buffer.add_subbytes into buf 0 n;
                      if Buffer.length into > limit then flooded := true
                  | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EINTR), _, _)
                    ->
                      ())
                readable;
              if writable <> [] then
                match
                  Unix.write_substring in_w input !sent
                    (String.length input - !sent)
                with
                | n ->
                    sent := !sent + n;
                    if !sent = String.length input then close_input ()
                | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EINTR), _, _)
                  ->
                    ()
                | exception Unix.Unix_error (Unix.EPIPE, _, _) -> close_input ()
              )
      done;
      (if !timed_out || !flooded then
         try Unix.kill (-pid) Sys.sigkill with Unix.Unix_error _ -> ());
      close_input ();
      List.iter Unix.close !reading;
      let _, status = Unix.waitpid [] pid in
      (* Whatever the session left behind. *)
      (try Unix.kill (-pid) Sys.sigkill with Unix.Unix_error _ -> ());
      (status, !timed_out, !flooded, Buffer.contents out, Buffer.contents err)

let payload (image : Coral_image.t) ~bound =
  let b = Buffer.create 4096 in
  List.iteri
    (fun k (s : Coral_layout.Placed_slot.t) ->
      if Coral_layout.Placement.read s.slot.placement then (
        let size = Int64.to_int s.slot.size in
        let bytes = Option.value (bound k) ~default:"" in
        let n = min size (String.length bytes) in
        Buffer.add_substring b bytes 0 n;
        Buffer.add_string b (String.make (size - n) '\000')))
    image.layout.slots;
  Buffer.contents b

(* The result payload cut back into the written slots, by slot index. *)
let outputs (image : Coral_image.t) data =
  let at = ref 0 in
  List.concat
    (List.mapi
       (fun k (s : Coral_layout.Placed_slot.t) ->
         if Coral_layout.Placement.written s.slot.placement then (
           let size = Int64.to_int s.slot.size in
           let bytes = String.sub data !at size in
           at := !at + size;
           [ (k, bytes) ])
         else [])
       image.layout.slots)

(* The RTL testbench's cycle limit and MPACT's [--max_instructions] stop. *)
let loop_limit = "loop limit"
let instruction_limit = "Stop reason: instruction_limit"

(* Simulator text allowed beyond the frame before a run counts as a flood. *)
let output_slack = 4 * 1024 * 1024

let contains s needle =
  let n = String.length needle in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = needle || go (i + 1))
  in
  go 0

(* The error for a simulator limit: the expectation that set it, if one did. *)
let limit_error budget plain =
  match budget with
  | Some (e, _) -> Run_error.Over_expectation e
  | None -> plain

(* A run may take twice what its model expects, plus what the runtime spends
   before and after the kernel. A model more than 200% short is a finding. *)
let fudge = 2L
let startup_slack = 100_000L
let limit_of expected = Int64.add (Int64.mul fudge expected) startup_slack

(* The simulator's own limit for an expectation: cycles on the RTL testbench,
   retired instructions on MPACT. An expectation of the other kind is ignored,
   so one call site may carry both. *)
let budget (profile : Coral_profile.t) ~label ~expect_cycles
    ~expect_instructions =
  let make kind expected flag =
    if Int64.compare expected 0L <= 0 then
      invalid_arg "Coral_run.run: an expectation must be positive";
    let limit = limit_of expected in
    Some (Expectation.{ expected; kind; label; limit }, flag limit)
  in
  match (profile.simulator, expect_cycles, expect_instructions) with
  | Rtl, Some e, _ -> make Cycles e (Printf.sprintf "--cycles=%Ld")
  | Mpact, _, Some e ->
      make Instructions e (Printf.sprintf "--max_instructions=%Ld")
  | _ -> None

let run ?(deadline = 120.) ?(flags = []) ?label ?expect_cycles
    ?expect_instructions ~dir (image : Coral_image.t) ~bound =
  (* The child runs in its own directory. *)
  let dir =
    if Filename.is_relative dir then Filename.concat (Sys.getcwd ()) dir
    else dir
  in
  let work = fresh_dir label in
  let elf = Filename.concat work "guest.elf" in
  let budget =
    budget image.profile ~label ~expect_cycles ~expect_instructions
  in
  let old = Sys.signal Sys.sigpipe Sys.Signal_ignore in
  Fun.protect
    ~finally:(fun () ->
      Sys.set_signal Sys.sigpipe old;
      (try Sys.remove elf with Sys_error _ -> ());
      try Sys.rmdir work with Sys_error _ -> ())
    (fun () ->
      Out_channel.with_open_bin elf (fun oc -> output_string oc image.elf);
      let argv =
        match Coral_profile.argv ~dir ~elf image.profile with
        | program :: args ->
            (* the budget first, then the manifest's own arguments (MPACT's
               carry a default instruction limit), then the caller's flags:
               the last of a repeated flag wins *)
            program :: (Option.to_list (Option.map snd budget) @ args @ flags)
        | [] -> []
      in
      let program = List.hd argv in
      if not (Sys.file_exists program) then
        Err.fail (Run_error.Simulator_missing program)
      else
        let input =
          Coral_wire.input ~digest:image.layout.digest (payload image ~bound)
        in
        let payload_size =
          List.fold_left
            (fun n (s : Coral_layout.Placed_slot.t) ->
              if Coral_layout.Placement.written s.slot.placement then
                Int64.add n s.slot.size
              else n)
            0L image.layout.slots
        in
        let limit = Int64.to_int payload_size + output_slack in
        let t0 = ref 0. in
        let status, timed_out, flooded, out, err =
          Coral_cpus.with_cpus image.Coral_image.profile.Coral_profile.threads
            (fun () ->
              t0 := Unix.gettimeofday ();
              spawn ~cwd:work ~deadline ~limit argv input)
        in
        let wall = Unix.gettimeofday () -. !t0 in
        if flooded then Err.fail (Run_error.Output_limit (Int64.of_int limit))
        else
          match
            Coral_wire.scan ~digest:image.layout.digest ~payload_size out
          with
          | Ok
              {
                frame = Result { status = st; mcycle; minstret; payload };
                before;
                after;
              } ->
              Err.return
                {
                  argv;
                  mcycle;
                  minstret;
                  outputs = outputs image payload;
                  status = st;
                  text = before ^ after ^ err;
                  wall;
                }
          | Ok { frame = Trap { mcause; mepc; mtval }; _ } -> (
              match
                if mcause = Coral_runtime.refused then
                  Coral_runtime.Refusal.of_code mepc
                else None
              with
              | Some r -> Err.fail (Run_error.Input_refused r)
              | None ->
                  Err.fail (Run_error.Guest_trap { Trap.mcause; mepc; mtval }))
          | Error _ when contains out loop_limit || contains err loop_limit ->
              Err.fail (limit_error budget Run_error.Cycle_limit)
          | Error _
            when contains out instruction_limit
                 || contains err instruction_limit ->
              Err.fail (limit_error budget Run_error.Instruction_limit)
          | Error _ when timed_out -> Err.fail Run_error.Timeout
          | Error Coral_wire.Malformed.No_frame when status <> Unix.WEXITED 0 ->
              Err.fail
                (Run_error.Simulator_crashed
                   { status = describe status; tail = tail (out ^ err) })
          | Error malformed ->
              Err.fail
                (Run_error.Malformed_output
                   { malformed; tail = tail (out ^ err) }))
