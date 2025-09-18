"""Pipeline task used for debugging and environment checks."""

import numpy as np
import yaml

from ... import config
from ..manager import TaskBase
from ._base import MPILoggedTask, SetMPILogging, SingleTask


class CheckMPIEnvironment(MPILoggedTask):
    """Check that the current MPI environment can communicate across all nodes."""

    timeout = config.Property(proptype=int, default=240)

    def setup(self):
        """Send random messages between all ranks.

        Tests to ensure that all messages are received within a specified amount
        of time, and that the messages received are the same as those sent (i.e.
        nothing was corrupted).
        """
        import time

        comm = self.comm
        n = 500000  # Corresponds to a 4 MB buffer
        results = []

        sends = np.arange(comm.size * n, dtype=np.float64).reshape(comm.size, n)
        recvs = np.empty_like(sends)

        # Send and receive across all ranks
        for i in range(comm.size):
            send = (comm.rank + i) % comm.size
            recv = (comm.rank - i) % comm.size

            results.append(comm.Irecv(recvs[recv, :], recv))
            comm.Isend(sends[comm.rank, :], send)

        start_time = time.time()

        while time.time() - start_time < self.timeout:
            success = all(r.get_status() for r in results)

            if success:
                break

            time.sleep(5)

        if not success:
            self.log.critical(
                f"MPI test failed to respond in {self.timeout} seconds. Aborting..."
            )
            comm.Abort()

        if not (recvs == sends).all():
            self.log.critical("MPI test did not receive the correct data. Aborting...")
            comm.Abort()

        # Stop successful processes from finshing if any task has failed
        comm.Barrier()

        self.log.debug(
            f"MPI test successful after {time.time() - start_time:.1f} seconds"
        )


class DebugInfo(MPILoggedTask, SetMPILogging):
    """Output some useful debug info."""

    def __init__(self):
        import logging

        # Set the default log levels to something reasonable for debugging
        self.level_rank0 = logging.DEBUG
        self.level_all = logging.INFO
        SetMPILogging.__init__(self)
        MPILoggedTask.__init__(self)

        ip = self._get_external_ip()

        self.log.info(f"External IP is {ip}")

        if self.comm.rank == 0:
            versions = self._get_package_versions()

            for name, version in versions:
                self.log.info(f"Package: {name:40s} version={version}")

    def _get_external_ip(self) -> str:
        # Reference here:
        # https://community.cloudflare.com/t/can-1-1-1-1-be-used-to-find-out-ones-public-ip-address/14971/6

        # Setup a resolver to point to Cloudflare
        import dns.resolver

        r = dns.resolver.Resolver()
        r.nameservers = ["1.1.1.1"]

        # Get IP from cloudflare chaosnet TXT record, and parse the response
        res = r.resolve("whoami.cloudflare", "TXT", "CH", tcp=True, lifetime=15)

        return str(res[0]).replace('"', "")

    def _get_package_versions(self) -> list[tuple[str, str]]:
        import json
        import subprocess

        p = subprocess.run(["pip", "list", "--format", "json"], stdout=subprocess.PIPE)

        package_info = json.loads(p.stdout)

        package_list = []

        for p in package_info:
            package_list.append((p["name"], p["version"]))

        return package_list


class Print(TaskBase):
    """Stupid module which just prints whatever it gets. Good for debugging."""

    def next(self, input_):
        """Print the input."""
        print(input_)

        return input_


class SaveModuleVersions(SingleTask):
    """Write module versions to a YAML file.

    The list of modules should be added to the configuration under key 'save_versions'.
    The version strings are written to a YAML file.

    Attributes
    ----------
    root : str
        Root of the file name to output to.
    """

    root = config.Property(proptype=str)

    done = True

    def setup(self):
        """Save module versions."""
        fname = f"{self.root}_versions.yml"
        f = open(fname, "w")
        f.write(yaml.safe_dump(self.versions))
        f.close()
        self.done = True

    def process(self):
        """Do nothing."""
        self.done = True
        return


class SaveConfig(SingleTask):
    """Write pipeline config to a text file.

    Yaml configuration document is written to a text file.

    Attributes
    ----------
    root : str
        Root of the file name to output to.
    """

    root = config.Property(proptype=str)
    done = True

    def setup(self):
        """Save module versions."""
        fname = f"{self.root}_config.yml"
        f = open(fname, "w")
        f.write(yaml.safe_dump(self.pipeline_config))
        f.close()
        self.done = True

    def process(self):
        """Do nothing."""
        self.done = True
        return
