(* The contraction helpers an artifact calls, as host functions: the AArch64
   route generates no code for a contraction, so each descriptor's loop nest is
   C built by the host compiler, loaded into the process, and bound to the
   artifact's [mlt_contract_*] symbol by a Rivet trampoline. The code under test
   is then everything around the contraction: the driver's loops, the address
   arithmetic of every base, and the calls. The sum is the descriptor's ordered
   one: each product and each sum rounded to binary32, no contraction. *)

module Art = Machine_model.Mir_artifact
module Proc = Loop_c_exec.Proc

let c_source (d : Contraction.t) =
  let open Contraction in
  let seed = match d.seed with Seed.Zero -> "0.0f" | Seed.Tensor -> "c[at]" in
  Fmt.str
    {|void %s(const float *a, const float *b, float *c) {
  for (long bb = 0; bb < %Ld; bb++)
    for (long i = 0; i < %Ld; i++)
      for (long j = 0; j < %Ld; j++) {
        long at = bb * %Ld + i * %Ld + j * %Ld;
        float acc = %s;
        for (long k = 0; k < %Ld; k++) {
          volatile float p = a[bb * %Ld + i * %Ld + k * %Ld] * b[bb * %Ld + j * %Ld + k * %Ld];
          acc = acc + p;
        }
        c[at] = acc;
      }
}
|}
    (symbol d) d.batch d.m d.n d.c.Output.batch d.c.Output.row d.c.Output.col
    seed d.k d.a.Operand.batch d.a.Operand.outer d.a.Operand.k d.b.Operand.batch
    d.b.Operand.outer d.b.Operand.k

let supported (d : Contraction.t) =
  d.Contraction.format = Contraction.Element.F32
  && d.Contraction.numerics = Contraction.Numerics.Ordered

let dir =
  lazy
    (let d = Proc.temp_dir "contract_host" in
     at_exit (fun () -> Proc.remove_tree d);
     d)

let counter = ref 0

let generated artifact =
  List.filter_map
    (fun (s : Art.Symbol.t) ->
      match s.Art.Symbol.kind with
      | Art.Symbol.Generated_function d -> Some d
      | _ -> None)
    (Art.symbols artifact)

(* The trampoline module binding every contraction the artifact calls. *)
let extra artifact =
  match generated artifact with
  | [] -> Ok []
  | ds when not (List.for_all supported ds) ->
      Error "a contraction the host helper does not implement"
  | ds -> (
      let dir = Lazy.force dir in
      (* a fresh name each time: dlopen of a loaded path returns that library *)
      incr counter;
      let c = Filename.concat dir (Fmt.str "contract%d.c" !counter) in
      let so = Filename.concat dir (Fmt.str "contract%d.so" !counter) in
      Proc.write_file c (String.concat "\n" (List.map c_source ds));
      match
        Proc.run
          (Loop_c_exec.Host.default_compiler
          @ [ "-O1"; "-ffp-contract=off"; "-shared"; "-fPIC"; c; "-o"; so ])
      with
      | Error m -> Error m
      | Ok (Proc.Exited 0, _) -> (
          let lib = Dl.dlopen ~filename:so ~flags:[ Dl.RTLD_NOW ] in
          let host_symbol name =
            match Dl.dlsym ~handle:lib ~symbol:name with
            | p -> Some (Int64.of_nativeint p)
            | exception _ -> None
          in
          match
            Err.payload
              (Machine_rivet_aarch64.Rivet_a64_module.helpers ~host_symbol
                 (List.map Contraction.symbol ds))
          with
          | Ok m -> Ok [ m ]
          | Error r ->
              Error (Fmt.str "%a" Machine_rivet_aarch64.Rivet_a64_refusal.pp r))
      | Ok (_, log) -> Error log)
