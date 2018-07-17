The purpose of this application is threefold:

1. To provide a unifying set of HTTP APIs to wrap all of the backend services regardless of their implementation (flask, ce-store, other)
2. For the unified APIs to present a large series of entrypoints from which any view on the system can be obtained.  This will be very useful for the conversational interaction.
3. To provide a simple "wireframe" User Interface to exercise the unified APIs and show the results in simple html pages

The intention is that the UI elements can either easily be reskinned (using CSS) to become the final UI, or these wireframes can be used a prototypes to inform a separate build of different UI components.