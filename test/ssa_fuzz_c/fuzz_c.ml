open Ssa_fuzz.Fuzz

let backend =
  let run case =
    let p = case.program in
    match Ssa_c.kernel ~name:Loop_c_exec.kernel_name p with
    | Error e -> Refused (Fmt.str "%a" Ssa_c.pp_error e)
    | Ok (kernel, sites) -> (
        let loop, bind = emitted case (Ssa_c.arguments p) in
        match
          Err.payload (Loop_c_exec.exec_kernel ~kernel ~sites loop ~bind)
        with
        | Error (#Loop_ir.Loop_interp.error as e) -> of_row e
        | Error e -> Fmt.failwith "%a" Loop_c_exec.pp_error e
        | Ok outputs -> of_outputs outputs)
  in
  { name = "c"; run }

let time_cap_s = 300.

let timed f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  let dt = Unix.gettimeofday () -. t0 in
  if dt > time_cap_s then Fmt.failwith "over the time cap: %.1fs" dt;
  r

let fail_on = function
  | Ok _ -> ()
  | Error d -> Fmt.failwith "%a" pp_disagreement d
