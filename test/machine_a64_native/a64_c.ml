(* The generated C harness: one function per form instance running exactly
   that instruction between a seeded NZCV write and an NZCV read, the table of
   them, an ADRP probe against a real symbol, and a main loop that fixes FPCR
   for the run and restores it. *)

open A64_forms

(* An assembly template as the body of a C string literal. *)
let c_escape s =
  String.concat ""
    (List.map
       (function
         | '\n' -> "\\n" | '\t' -> "\\t" | '"' -> "\\\"" | c -> String.make 1 c)
       (List.of_seq (String.to_seq s)))

let c_function k (f : Form.t) =
  let decl i r =
    let n = List.nth names i in
    match r with
    | G _ -> Printf.sprintf "  uint64_t %s = in[%d];" n i
    | F _ -> Printf.sprintf "  v128 %s = { in[%d], 0 };" n i
  in
  let input i r =
    let n = List.nth names i in
    match r with
    | G _ -> Printf.sprintf "[%s] \"r\" (%s)" n n
    | F _ -> Printf.sprintf "[%s] \"w\" (%s)" n n
  in
  let result_decl, result_out, result_store =
    match f.Form.result with
    | Some (G _) ->
        ("  uint64_t d = seed;", "[d] \"+r\" (d)", "  o[0] = d; o[1] = 0;")
    | Some (F _) ->
        ( "  v128 d = { seed, seed };",
          "[d] \"+w\" (d)",
          "  o[0] = d[0]; o[1] = d[1];" )
    | None -> ("", "", "  o[0] = 0; o[1] = 0;")
  in
  let outs =
    String.concat ", "
      (List.filter (( <> ) "") [ result_out; "[flags] \"=&r\" (flags)" ])
  in
  let ins =
    String.concat ", "
      (List.mapi input f.Form.inputs
      @ [ "[nz] \"r\" (nz)" ]
      @ if f.Form.mem then [ "[base] \"r\" (buf + 8)" ] else [])
  in
  String.concat "\n"
    ([
       Printf.sprintf
         "static void form_%d(const uint64_t *in, uint64_t nz, uint64_t seed, \
          uint8_t *buf, uint64_t *o) {"
         k;
     ]
    @ List.mapi decl f.Form.inputs
    @ [
        result_decl;
        "  uint64_t flags;";
        "  __asm__ volatile(";
        "    \"msr nzcv, %x[nz]\\n\\t\"";
        Printf.sprintf "    \"%s\\n\\t\"" (c_escape f.Form.asm);
        "    \"mrs %x[flags], nzcv\"";
        Printf.sprintf "    : %s" outs;
        Printf.sprintf "    : %s" ins;
        "    : \"cc\", \"memory\");";
        result_store;
        "  o[2] = flags;";
        "}";
      ])

let c_program () =
  String.concat "\n"
    ([
       "#ifndef __aarch64__";
       "#error \"the AArch64 conformance harness needs an AArch64 compiler\"";
       "#endif";
       "#include <stdint.h>";
       "#include <stdio.h>";
       "#include <string.h>";
       "#include <sys/auxv.h>";
       "#include <asm/hwcap.h>";
       "typedef uint64_t v128 __attribute__((vector_size(16)));";
     ]
    @ List.mapi c_function forms
    @ [
        "typedef void (*form_fn)(const uint64_t *, uint64_t, uint64_t, uint8_t \
         *, uint64_t *);";
        Printf.sprintf "static const form_fn table[] = { %s };"
          (String.concat ", "
             (List.mapi (fun k _ -> Printf.sprintf "form_%d" k) forms));
        "static uint8_t probe_sym[3 * 4096] __attribute__((aligned(4096)));";
        "/* ADRP and ADD :lo12: against a real symbol: the page of the target \
         and the target */";
        "#define PROBE(K) do { uint64_t page, full; __asm__ volatile(\"adrp \
         %0, probe_sym+\" #K \"\\n\\tadd %1, %0, #:lo12:probe_sym+\" #K : \
         \"=&r\"(page), \"=r\"(full)); printf(\"addr %d %llx %llx %llx\\n\", \
         K, (unsigned long long)(uintptr_t)(probe_sym + K), (unsigned long \
         long)page, (unsigned long long)full); } while (0)";
        "static void probe_addresses(void) { PROBE(0); PROBE(8); PROBE(4095); \
         PROBE(4112); PROBE(8191); }";
        "int main(void) {";
        "  unsigned long hw = getauxval(AT_HWCAP);";
        "  if (!(hw & HWCAP_FP) || !(hw & HWCAP_ASIMD)) { puts(\"unavailable \
         fp/asimd\"); return 2; }";
        "  uint64_t fpcr_saved, fpcr;";
        "  __asm__ volatile(\"mrs %0, fpcr\" : \"=r\"(fpcr_saved));";
        "  /* round to nearest even, no flush to zero, no default NaN, no \
         traps */";
        "  __asm__ volatile(\"msr fpcr, %0\" :: \"r\"((uint64_t)0));";
        "  __asm__ volatile(\"mrs %0, fpcr\" : \"=r\"(fpcr));";
        "  printf(\"fpcr %llx\\n\", (unsigned long long)fpcr);";
        "  probe_addresses();";
        "  unsigned k; unsigned long long a, b, c, nz, seed, m[4];";
        "  while (scanf(\"%u %llx %llx %llx %llx %llx %llx %llx %llx %llx\", \
         &k, &a, &b, &c, &nz, &seed, &m[0], &m[1], &m[2], &m[3]) == 10) {";
        "    uint64_t in[3] = { a, b, c }, o[3]; uint8_t buf[32];";
        "    memcpy(buf, m, 32);";
        "    table[k](in, nz, seed, buf, o);";
        "    memcpy(m, buf, 32);";
        "    printf(\"%llx %llx %llx %llx %llx %llx %llx\\n\", (unsigned long \
         long)o[0], (unsigned long long)o[1], (unsigned long long)o[2], m[0], \
         m[1], m[2], m[3]);";
        "  }";
        "  __asm__ volatile(\"msr fpcr, %0\" :: \"r\"(fpcr_saved));";
        "  return 0;";
        "}";
        "";
      ])
