import React, { useEffect, useState } from 'react';
import { Box, Typography, Container, Paper } from '@mui/material';
import axios from 'axios';

const VehicleCountPage = () => {
    const [vehicleCount, setVehicleCount] = useState(0);

    useEffect(() => {
        const fetchVehicleCount = async () => {
            const response = await axios.get('/api/predict-vehicles');
            setVehicleCount(response.data.count);
        };

        const interval = setInterval(fetchVehicleCount, 3000);
        return () => clearInterval(interval);
    }, []);

    return (
        <Container maxWidth="lg">
            <Box
                sx={{
                    marginTop: 8,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                }}
            >
                <Typography variant="h4" gutterBottom>
                    Vehicle Tracking
                </Typography>
                <Paper
                    sx={{
                        padding: 4,
                        marginTop: 4,
                        width: '100%',
                        maxWidth: 600,
                        textAlign: 'center',
                    }}
                >
                    <Typography variant="h5" color="primary">
                        Current Vehicle Count: {vehicleCount}
                    </Typography>
                </Paper>
            </Box>
        </Container>
    );
};

export default VehicleCountPage;