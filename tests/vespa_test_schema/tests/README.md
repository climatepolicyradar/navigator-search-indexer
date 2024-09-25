## Vespa tests

Details on different kinds of tests (system, staging & production) can be found in the [vespa documentation](https://docs.vespa.ai/en/reference/testing.html)

These are declared by defining specific subfolder within this app specification. It's also possible to write custom tests using Java if we ever feel like doing that.

Tests run on deploys, you can also run an individual test locally via the test command:

```
vespa test -i <instance_to_run_against> navigator_app/tests/system-test/hybrid-search-test.json
```
