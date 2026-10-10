open Transformers_metadata.Json_util
open World
open Pt2_fixture_unix_test.Support

let%expect_test
    "model execution restores checkpoint pins absent from metadata-only task \
     references" =
  with_dir (fun dir ->
      let fixture = build () in
      let calls = ref [] in
      let config = online fixture dir calls in
      let bundle = get (T.Reference.ensure config cohort fixture.request) in
      let reference = List.hd bundle.references in
      Printf.printf "metadata sources=%d, checkpoint downloaded=%b\n"
        (List.length reference.bundle.entry.sources)
        (List.mem (W.url R.source_name) !calls);
      Printf.printf "metadata execution refused=%b\n"
        (Result.is_error
           (Err.payload (U.Fixture.of_bundle config reference.bundle)));
      let opened = get (T.Demo.execution_fixture config cohort reference) in
      Printf.printf "execution sources=%d, checkpoint downloaded=%b\n"
        (List.length opened.bundle.entry.sources)
        (List.mem (W.url R.source_name) !calls);
      let contract = get (text reference.contract >>= F.Contract.of_string) in
      let rows = get (member "cases" bundle.manifest >>= array) in
      List.iter
        (fun row ->
          let inputs = get (T.Adapter.role bundle row "inputs") in
          let expected = get (T.Adapter.role bundle row "published") in
          let actual = get (T.Input.run opened.archive contract inputs) in
          let checks =
            get
              (T.Diagnostic.compare ~atol:contract.atol ~rtol:contract.rtol
                 expected actual)
          in
          Printf.printf "%s outputs=%d all passed=%b\n"
            (get (field "id" row))
            (List.length checks)
            (List.for_all F.Compare.passed checks))
        rows;
      let offline = U.Bundle.config config.cache in
      let repeated = get (T.Demo.execution_fixture offline cohort reference) in
      Printf.printf "offline sources=%d\n"
        (List.length repeated.bundle.entry.sources));
  [%expect
    {|
  metadata sources=0, checkpoint downloaded=false
  metadata execution refused=true
  execution sources=1, checkpoint downloaded=true
  reference-00-case-00 outputs=2 all passed=true
  reference-00-case-01 outputs=2 all passed=true
  offline sources=1 |}]
